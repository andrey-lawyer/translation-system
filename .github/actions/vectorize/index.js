/// .github/actions/vectorize/index.js
import fs from "fs";
import path from "path";
import { ChromaClient } from "chromadb";
import {  pipeline } from "@xenova/transformers";

// --- Настройка Chroma Cloud ---
const CHROMA_API_URL = "https://api.trychroma.com:8000";
const CHROMA_API_KEY = process.env.CHROMA_API_KEY; // ключ в секретах GitHub Actions
const COLLECTION_NAME = "translation_vectors";

const client = new ChromaClient({
    apiUrl: CHROMA_API_URL,
    apiKey: CHROMA_API_KEY,
});

async function getOrCreateCollection(name) {
    try {
        return await client.getCollection(name);
    } catch (err) {
        if (err.name === "ChromaUniqueError") {
            console.log("Collection already exists, using it...");
            return await client.getCollection(name);
        } else {
            return await client.createCollection({ name });
        }
    }
}

// --- Загрузка локальной модели для эмбеддингов ---
console.log("Loading local embedding model...");
const embedPipeline = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
console.log("Model loaded ✅");

// --- Получение файлов для векторизации ---
const ROOT_DIR = path.resolve(process.cwd());
const FILES = [
    ".github/actions/vectorize/index.js",
    "README.MD",
    "go-translator/internal/server/server.go",
    "go-translator/internal/translator/myMemory.go",
    "go-translator/internal/translator/service.go",
    "react-front/src/App.js",
];

async function embedFile(filePath) {
    const content = fs.readFileSync(path.resolve(ROOT_DIR, filePath), "utf-8");
    const chunk = content.substring(0, 1000); // для примера, берем первые 1000 символов
    const vector = await embedPipeline(chunk);
    return {
        id: filePath,
        vector,
        metadata: { file: filePath },
    };
}

async function main() {
    const collection = await getOrCreateCollection(COLLECTION_NAME);

    console.log("Starting vectorization...");
    for (const file of FILES) {
        try {
            const doc = await embedFile(file);
            await collection.add([doc]);
            console.log(`✅ Embedded ${file}`);
        } catch (err) {
            console.error(`Error embedding ${file}:`, err.message);
        }
    }
    console.log("Vectorization complete");
}

main().catch(console.error);



