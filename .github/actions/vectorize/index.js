import { pipeline } from "@xenova/transformers";
import { CloudClient } from "chromadb";
import fs from "fs/promises";

// Chroma Cloud настройки из env
const CHROMA_API_KEY = process.env.CHROMA_API_KEY || '';
const CHROMA_TENANT = process.env.CHROMA_TENANT || '';
const CHROMA_DATABASE = process.env.CHROMA_DATABASE || '';
const COLLECTION_NAME = process.env.COLLECTION_NAME || 'TestCollection';

// Файлы для векторизации
const filesToVectorize = [
    ".github/actions/vectorize/index.js",
    "README.MD",
    "go-translator/internal/server/server.go",
    "react-front/src/App.js",
];

// Функция усреднения токенов в один embedding
function flattenEmbedding(embedding) {
    if (!Array.isArray(embedding)) return [];
    if (embedding.length === 0) return [];
    if (Array.isArray(embedding[0])) {
        const length = embedding[0].length;
        const sum = new Array(length).fill(0);
        for (const tokenVec of embedding) {
            for (let i = 0; i < length; i++) sum[i] += tokenVec[i] || 0;
        }
        return sum.map(x => x / embedding.length);
    }
    return embedding;
}

async function getOrCreateCollection(client, name) {
    try {
        console.log(`⏳ Ищем коллекцию: ${name}...`);
        const collection = await client.getCollection({ name });
        console.log(`✅ Коллекция готова: ID=${collection.id}, name=${collection.name}`);
        return collection;
    } catch (err) {
        if (err.message.includes("collection not found") || err.name === "ChromaConnectionError") {
            console.log(`Создаём новую коллекцию: ${name}...`);
            const collection = await client.createCollection({ name });
            console.log(`✅ Коллекция создана: ID=${collection.id}, name=${collection.name}`);
            return collection;
        }
        console.error("❌ Ошибка получения коллекции:", err.message);
        return null;
    }
}

async function main() {
    console.log("⏳ Загружаем локальную модель эмбеддингов...");
    const embedder = await pipeline("feature-extraction");
    console.log("✅ Модель загружена");

    // Генерация эмбеддингов
    const embeddings = {};
    for (const file of filesToVectorize) {
        try {
            const text = await fs.readFile(file, "utf-8");
            const embeddingRaw = await embedder(text);
            const embedding = flattenEmbedding(embeddingRaw);

            if (!embedding || embedding.length === 0) {
                console.warn(`⚠️ Empty embedding for ${file}, пропускаем.`);
                continue;
            }

            embeddings[file] = embedding;
            console.log(`✅ Embedded ${file} (length: ${embedding.length})`);
        } catch (err) {
            console.error(`❌ Failed embedding ${file}:`, err.message);
        }
    }

    // Загрузка в Chroma Cloud
    if (CHROMA_API_KEY && CHROMA_TENANT && CHROMA_DATABASE) {
        try {
            console.log("⏳ Подключаемся к Chroma Cloud...");
            const client = new CloudClient({
                apiKey: CHROMA_API_KEY,
                tenant: CHROMA_TENANT,
                database: CHROMA_DATABASE,
            });

            const collection = await getOrCreateCollection(client, COLLECTION_NAME);
            if (!collection) {
                console.warn("⚠️ Коллекция недоступна. Пропускаем загрузку в Chroma Cloud.");
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
        } catch (err) {
            console.error("❌ Chroma Cloud connection failed:", err.message);
        }
    } else {
        console.log("⚠️ Chroma Cloud credentials not provided, пропускаем загрузку.");
    }

    console.log("🎯 Vectorization complete ✅");
}

main().catch(err => console.error("💥 Fatal error:", err));

