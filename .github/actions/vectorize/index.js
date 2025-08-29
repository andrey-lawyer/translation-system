// .github/actions/vectorize/index.js
import { pipeline } from "@xenova/transformers";
import { CloudClient } from "chromadb";

const CHROMA_API_KEY = process.env.CHROMA_API_KEY || '';
const CHROMA_TENANT = process.env.CHROMA_TENANT || '';
const CHROMA_DATABASE = process.env.CHROMA_DATABASE || '';
const COLLECTION_NAME = "TestCollection";

async function getOrCreateCollection(client, name) {
    try {
        // пытаемся получить коллекцию
        const collection = await client.getCollection(name);
        return collection;
    } catch (err) {
        // если коллекция не найдена, создаём
        if (err.message.includes("collection not found") || err.name === "ChromaConnectionError") {
            try {
                console.log("Создаём новую коллекцию:", name);
                return await client.createCollection({ name });
            } catch (createErr) {
                console.error("Не удалось создать коллекцию:", createErr.message);
                return null;
            }
        }
        console.error("Ошибка получения коллекции:", err.message);
        return null;
    }
}

async function main() {
    console.log("Loading local embedding model...");
    const embedder = await pipeline("feature-extraction");
    console.log("Model loaded ✅");

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
            const text = await (await import('fs/promises')).readFile(file, "utf-8");
            const embedding = await embedder(text);
            embeddings[file] = embedding;
            console.log(`✅ Embedded ${file}`);
        } catch (err) {
            console.error(`❌ Failed embedding ${file}:`, err.message);
        }
    }

    // Пытаемся подключиться к Chroma Cloud (опционально)
    if (CHROMA_API_KEY && CHROMA_TENANT && CHROMA_DATABASE) {
        try {
            console.log("Connecting to Chroma Cloud...");
            const client = new CloudClient({
                apiKey: CHROMA_API_KEY,
                tenant: CHROMA_TENANT,
                database: CHROMA_DATABASE,
            });

            const collection = await getOrCreateCollection(client, COLLECTION_NAME);
            if (!collection) {
                console.warn("Коллекция недоступна. Пропускаем загрузку в Chroma Cloud.");
                return;
            }

            // Пушим эмбеддинги в Chroma
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
            console.error("Chroma Cloud connection failed:", err.message);
        }
    } else {
        console.log("Chroma Cloud credentials not provided, skipping upload.");
    }

    console.log("Vectorization complete ✅");
}

main().catch(err => console.error("Fatal error:", err));




