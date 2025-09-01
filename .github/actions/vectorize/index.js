/// .github/actions/vectorize/index.js
import { pipeline } from "@xenova/transformers";
import { CloudClient } from "chromadb";
import fs from "fs/promises";

// --- ENV ---
const CHROMA_API_KEY = process.env.CHROMA_API_KEY || "";
const CHROMA_TENANT = process.env.CHROMA_TENANT || "";
const CHROMA_DATABASE = process.env.CHROMA_DATABASE || "";
const COLLECTION_NAME = process.env.COLLECTION_NAME || "Test";

// --- Получение/создание коллекции ---
async function getOrCreateCollection(client, name) {
    try {
        console.log("DEBUG - All env vars:", {
            CHROMA_API_KEY: CHROMA_API_KEY ? "[set]" : "[not set]",
            CHROMA_TENANT: CHROMA_TENANT ? "[set]" : "[not set]",
            CHROMA_DATABASE: CHROMA_DATABASE ? "[set]" : "[not set]",
            COLLECTION_NAME: name,
        });

        const collection = await client.getCollection({ name });
        console.log("✅ Коллекция найдена:", name);
        return collection;
    } catch (err) {
        if (err.message?.includes("not found")) {
            console.log("ℹ️ Коллекция не найдена. Создаём новую:", name);
            const collection = await client.createCollection({ name });
            console.log("✅ Создана новая коллекция:", name);
            return collection;
        }
        console.error("❌ Ошибка при получении коллекции:", err.message);
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
    ];

    // --- Эмбеддинги ---
    const embeddings = {};
    for (const file of filesToVectorize) {
        try {
            const text = await fs.readFile(file, "utf-8");
            const embedding = await embedder(text);
            embeddings[file] = embedding;
            console.log(`✅ Embedded ${file}`);
        } catch (err) {
            console.error(`❌ Ошибка эмбеддинга ${file}:`, err.message);
        }
    }

    // --- Подключение к Chroma Cloud ---
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
                console.warn("⚠️ Коллекция недоступна. Пропускаем загрузку.");
                return;
            }

            // --- Загрузка эмбеддингов в Chroma ---
            for (const [file, vector] of Object.entries(embeddings)) {
                try {
                    await collection.add({
                        ids: [file],
                        embeddings: [vector],
                        metadatas: [{ file }],
                    });
                    console.log(`✅ Загружен ${file} в Chroma Cloud`);
                } catch (err) {
                    console.error(`❌ Ошибка загрузки ${file}:`, err.message);
                }
            }
        } catch (err) {
            console.error("❌ Ошибка подключения к Chroma Cloud:", err.message);
        }
    } else {
        console.log("⚠️ Данные Chroma Cloud не заданы. Пропускаем загрузку.");
    }

    console.log("🎉 Векторизация завершена");
}

main().catch((err) => console.error("🔥 Fatal error:", err));





