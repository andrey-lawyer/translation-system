import { pipeline } from "@xenova/transformers";
import { CloudClient } from "chromadb";
import fs from "fs/promises";

const CHROMA_API_KEY = process.env.CHROMA_API_KEY || '';
const CHROMA_TENANT = process.env.CHROMA_TENANT || '';
const CHROMA_DATABASE = process.env.CHROMA_DATABASE || '';
const COLLECTION_NAME = process.env.COLLECTION_NAME || 'TestCollection';

async function getOrCreateCollection(client, name) {
    try {
        console.log(`⏳ Ищем коллекцию: ${name}...`);
        const collection = await client.getOrCreateCollection({ name });
        console.log(`✅ Коллекция готова: ID=${collection.id}, name=${collection.name}`);
        return collection;
    } catch (err) {
        console.error("❌ Ошибка при получении/создании коллекции:", err.message);
        return null;
    }
}

async function main() {
    console.log("⏳ Загружаем локальную модель эмбеддингов...");
    const embedder = await pipeline("feature-extraction");
    console.log("✅ Модель загружена");

    const filesToVectorize = [
        ".github/actions/vectorize/index.js",
        "README.MD",
        "go-translator/internal/server/server.go",
        "react-front/src/App.js",
        // добавь остальные файлы по необходимости
    ];

    // Генерируем эмбеддинги локально
    const embeddings = {};
    for (const file of filesToVectorize) {
        try {
            const text = await fs.readFile(file, "utf-8");
            const embedding = await embedder(text);
            embeddings[file] = embedding;
            console.log(`✅ Embedded ${file}`);
        } catch (err) {
            console.error(`❌ Failed embedding ${file}:`, err.message);
        }
    }

    // Проверяем, есть ли все переменные для Chroma Cloud
    console.log("DEBUG - Env vars:", {
        CHROMA_API_KEY: CHROMA_API_KEY ? "[set]" : "[not set]",
        CHROMA_TENANT: CHROMA_TENANT ? "[set]" : "[not set]",
        CHROMA_DATABASE: CHROMA_DATABASE ? "[set]" : "[not set]",
        COLLECTION_NAME
    });

    if (CHROMA_API_KEY && CHROMA_TENANT && CHROMA_DATABASE) {
        try {
            console.log("⏳ Подключаемся к Chroma Cloud...");
            const client = new CloudClient({
                apiKey: CHROMA_API_KEY,
                tenant: CHROMA_TENANT,
                database: CHROMA_DATABASE,
                path: "https://api.trychroma.com" // правильный путь для облака
            });

            const collection = await getOrCreateCollection(client, COLLECTION_NAME);
            if (!collection) {
                console.warn("⚠️ Коллекция недоступна. Пропускаем загрузку в Chroma Cloud.");
                return;
            }

            // Пушим эмбеддинги
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
            console.error("❌ Ошибка подключения к Chroma Cloud:", err.message);
        }
    } else {
        console.log("⚠️ Chroma Cloud credentials not provided, skipping upload.");
    }

    console.log("Vectorization complete ✅");
}

main().catch(err => console.error("Fatal error:", err));






