// .github/actions/vectorize/index.js
import { pipeline } from "@xenova/transformers";
import { CloudClient } from "chromadb";

const CHROMA_API_KEY = process.env.CHROMA_API_KEY || '';
const CHROMA_TENANT = process.env.CHROMA_TENANT || '';
const CHROMA_DATABASE = process.env.CHROMA_DATABASE || '';
const COLLECTION_ID = process.env.COLLECTION_ID || '';

const COLLECTION_NAME = "TestCollection";


async function getOrCreateCollection(client, name) {
    try {
        console.log("COLLECTION_ID- " + COLLECTION_ID)
        return await client.getCollection({ collectionId: COLLECTION_ID });
    } catch (err) {
        if (err.message.includes("collection not found") || err.name === "ChromaConnectionError") {
            console.log("Создаём новую коллекцию:", name);
            return await client.createCollection({ name, collectionId: COLLECTION_ID });
        }
        console.error("Ошибка получения коллекции:", err.message);
        return null;
    }
}

async function main() {
    console.log("Loading local embedding model...");
    const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
        quantized: true,
        revision: 'main',
        device: 'cpu',
        pooling: 'mean',
        normalize: true
    });
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
            console.log(`Processing ${file}...`);
            const text = await (await import('fs/promises')).readFile(file, "utf-8");
            
            // Get embeddings with default settings
            const output = await embedder(text);
            
            // Convert output to array if it's a tensor
            let embedding;
            if (output && typeof output === 'object' && 'data' in output) {
                // Handle tensor output
                embedding = Array.from(output.data);
            } else if (Array.isArray(output)) {
                // Handle array output directly
                embedding = output.flat(Infinity);
            } else if (output && typeof output === 'object') {
                // Handle object with numeric keys
                embedding = Object.values(output);
            } else {
                console.warn(`Unexpected output format from embedder for ${file}`);
                continue;
            }
            
            if (!embedding || embedding.length === 0) {
                console.warn(`Empty embedding for ${file}`);
                continue;
            }
            
            console.log(`Generated embedding for ${file} (length: ${embedding.length})`);
            embeddings[file] = embedding;
            
        } catch (err) {
            console.error(`❌ Failed processing ${file}:`, err);
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




