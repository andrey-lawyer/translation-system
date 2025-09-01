import { pipeline } from "@xenova/transformers";
import { CloudClient } from "chromadb";
import fs from "fs/promises";

// Загружаем переменные окружения
const CHROMA_API_KEY = process.env.CHROMA_API_KEY || '';
const CHROMA_TENANT = process.env.CHROMA_TENANT || '';
const CHROMA_DATABASE = process.env.CHROMA_DATABASE || '';
const COLLECTION_NAME = process.env.COLLECTION_NAME || 'TestCollection';

async function getOrCreateCollection(client, name) {
    try {
        console.log(`⏳ Ищем коллекцию: ${name}...`);
        const collection = await client.getCollection({ name });
        console.log(`✅ Коллекция готова: ID=${collection.id}, name=${collection.name}`);
        return collection;
    } catch (err) {
        console.warn(`⚠️ Коллекция "${name}" не найдена или ошибка соединения.`);
        console.warn("DEBUG - Ошибка при получении коллекции:", err.message);
        try {
            console.log(`⏳ Создаём новую коллекцию: ${name}...`);
            const collection = await client.createCollection({ name });
            console.log(`✅ Коллекция создана: ID=${collection.id}, name=${collection.name}`);
            return collection;
        } catch (createErr) {
            console.error("❌ Не удалось создать коллекцию:", createErr.message);
            return null;
        }
    }
}

async function main() {
    console.log("⏳ Загружаем локальную модель эмбеддингов...");
    const embedder = await pipeline("feature-extraction");
    console.log("✅ Модель загружена");

    // Список файлов для векторизации
    const filesToVectorize = [
        ".github/actions/vectorize/index.js",
        "README.MD",
        "go-translator/internal/server/server.go",
        "react-front/src/App.js",
    ];

    // Генерация эмбеддингов локально
    const embeddings = {};
    for (const file of filesToVectorize) {
        try {
            const text = await fs.readFile(file, "utf-8");
            const rawEmbedding = await embedder(text);
            // Исправляем формат: если [[...]], берем первый элемент
            const embeddingVector = Array.isArray(rawEmbedding[0]) ? rawEmbedding[0] : rawEmbedding;
            embeddings[file] = embeddingVector;
            console.log(`✅ Embedded ${file}`);
        } catch (err) {
            console.error(`❌ Failed embedding ${file}:`, err.message);
        }
    }

    // Подключение к Chroma Cloud (опционально)
    if (CHROMA_API_KEY && CHROMA_TENANT && CHROMA_DATABASE) {
        console.log("⏳ Подключаемся к Chroma Cloud...");
        const client = new CloudClient({
            apiKey: CHROMA_API_KEY,
            tenant: CHROMA_TENANT,
            database: CHROMA_DATABASE,
        });

        const collection = await getOrCreateCollection(client, COLLECTION_NAME);
        if (!collection) {
            console.warn("⚠️ Коллекция недоступна. Пропускаем загрузку в Chroma Cloud.");
        } else {
            // Загружаем эмбеддинги
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
                    // Расширенный дебаг
                    if (err.response) {
                        try {
                            const text = await err.response.text();
                            console.error("DEBUG - Response body:", text);
                        } catch {}
                    }
                }
            }
        }
    } else {
        console.log("⚠️ Chroma Cloud credentials not provided, skipping upload.");
    }

    console.log("Vectorization complete ✅");
}

main().catch(err => console.error("Fatal error:", err));







