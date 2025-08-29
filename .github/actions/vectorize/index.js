import fs from "fs";
import path from "path";
import { CloudClient } from "chromadb";
import { pipeline } from "@xenova/transformers";

console.log("Loading local embedding model...");

// Локальная модель эмбеддингов
const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");

console.log("Model loaded ✅");

// Chroma Cloud
const client = new CloudClient({
    apiKey: process.env.CHROMA_API_KEY,
    tenant: process.env.CHROMA_TENANT_ID,
    database: "Test"
});

// Проверяем или создаём коллекцию
let collection;
// пытаемся создать коллекцию, игнорируя дубликаты
try {
    collection = await client.createCollection({
        name: collectionName,
        metadata: { source: "vectorize-action" },
    });
} catch (err) {
    if (err.name === "ChromaUniqueError") {
        // коллекция уже существует — ищем её по имени
        const allCollections = await client.listCollections();
        collection = allCollections.find(c => c.name === collectionName);
        if (!collection) throw new Error("Collection exists but not found!");
    } else {
        throw err;
    }
}

// Рекурсивно обходим репозиторий
async function walk(dir) {
    const files = fs.readdirSync(dir);
    for (const file of files) {
        const fullPath = path.join(dir, file);
        const stat = fs.statSync(fullPath);

        if (stat.isDirectory()) {
            await walk(fullPath);
        } else {
            const text = fs.readFileSync(fullPath, "utf-8");

            // Разбиваем на куски по ~500 слов
            const chunks = text.match(/(.|[\r\n]){1,2000}/g) || [];

            for (const chunk of chunks) {
                if (!chunk.trim()) continue; // пропускаем пустые строки

                try {
                    // Локальная эмбеддинг функция
                    const tokens = await embedder(chunk);
                    // усредняем токены в один вектор
                    const vector = tokens[0][0].map((_, i) =>
                        tokens[0].reduce((sum, t) => sum + t[i], 0) / tokens[0].length
                    );

                    // Добавляем в Chroma Cloud
                    await collection.add({
                        ids: [fullPath + Math.random().toString(36).substr(2, 5)],
                        embeddings: [Array.from(vector)],
                        metadatas: [{ file: fullPath }],
                        documents: [chunk]
                    });

                    console.log(`✅ Embedded ${fullPath}`);
                } catch (err) {
                    console.error(`Error embedding ${fullPath}:`, err.message);
                }
            }
        }
    }
}

console.log("Starting vectorization (local embeddings + Chroma Cloud)...");
await walk(process.cwd());
console.log("✅ Vectorization complete");


