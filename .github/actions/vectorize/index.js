import { pipeline } from "@xenova/transformers";
import { CloudClient } from "chromadb";
import fs from "fs/promises";

// Параметры Chroma
const CHROMA_API_KEY = process.env.CHROMA_API_KEY || '';
const CHROMA_TENANT = process.env.CHROMA_TENANT || '';
const CHROMA_DATABASE = process.env.CHROMA_DATABASE || '';
const COLLECTION_NAME = process.env.COLLECTION_NAME || 'TestCollection';

// Настройки разбиения текста
const CHUNK_SIZE = 1000; // символов на кусок

async function getOrCreateCollection(client, name) {
    try {
        console.log(`⏳ Ищем коллекцию: ${name}...`);
        const collection = await client.getCollection({ name });
        console.log(`✅ Коллекция готова: ID=${collection.id}, name=${collection.name}`);
        return collection;
    } catch (err) {
        console.warn(`⚠️ Коллекция не найдена, создаём новую: ${name}`);
        const collection = await client.createCollection({ name });
        console.log(`✅ Создана коллекция: ID=${collection.id}, name=${collection.name}`);
        return collection;
    }
}

function averageEmbeddings(chunks) {
    if (!chunks.length) return [];
    const dim = chunks[0].length;
    const avg = new Array(dim).fill(0);
    for (const chunk of chunks) {
        for (let i = 0; i < dim; i++) {
            avg[i] += chunk[i];
        }
    }
    return avg.map(v => v / chunks.length);
}

async function embedText(embedder, text) {
    const chunks = [];
    for (let i = 0; i < text.length; i += CHUNK_SIZE) {
        const slice = text.slice(i, i + CHUNK_SIZE);
        const embRaw = await embedder(slice);
        if (Array.isArray(embRaw[0]) && embRaw[0].length) {
            chunks.push(embRaw[0]);
        }
    }
    return averageEmbeddings(chunks);
}

async function main() {
    console.log("⏳ Загружаем локальную модель эмбеддингов...");
    const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
    console.log("✅ Модель загружена");

    const filesToVectorize = [
        ".github/actions/vectorize/index.js",
        "README.MD",
        "go-translator/internal/server/server.go",
        "react-front/src/App.js",
    ];

    const embeddings = {};

    for (const file of filesToVectorize) {
        try {
            const text = await fs.readFile(file, "utf-8");
            const embedding = await embedText(embedder, text);
            if (!embedding.length) {
                console.warn(`⚠️ Empty embedding for ${file}, пропускаем`);
                continue;
            }
            embeddings[file] = embedding;
            console.log(`✅ Embedded ${file} (length: ${embedding.length})`);
        } catch (err) {
            console.error(`❌ Failed embedding ${file}:`, err.message);
        }
    }

    if (CHROMA_API_KEY && CHROMA_TENANT && CHROMA_DATABASE) {
        console.log("⏳ Подключаемся к Chroma Cloud...");
        const client = new CloudClient({
            apiKey: CHROMA_API_KEY,
            tenant: CHROMA_TENANT,
            database: CHROMA_DATABASE,
        });

        const collection = await getOrCreateCollection(client, COLLECTION_NAME);
        if (!collection) {
            console.warn("⚠️ Коллекция недоступна. Пропускаем загрузку.");
            return;
        }

        for (const [file, vector] of Object.entries(embeddings)) {
            try {
                await collection.add({
                    ids: [file],
                    embeddings: [vector],
                    metadatas: [{ file }],
                });
                console.log(`✅ Pushed ${file} to Chroma Cloud`);
            } catch (err) {
                console.error(`❌ Failed to push ${file}:`, err.message);
            }
        }
    } else {
        console.log("ℹ️ Chroma Cloud credentials not provided, skipping upload.");
    }

    console.log("🎯 Vectorization complete ✅");
}

main().catch(err => console.error("Fatal error:", err));



