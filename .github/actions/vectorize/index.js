// index.js
import fs from "fs";
import path from "path";
import { ChromaClient } from "chromadb";
import { OpenAIEmbeddings } from "langchain/embeddings/openai"; // локальные эмбеддинги через OpenAI

// -------------------- Настройки --------------------

// Название коллекции Chroma
const collectionName = "translation_system";

// Путь к директории с файлами для векторизации
const DATA_DIR = path.resolve(process.cwd());

// -------------------- Инициализация клиента Chroma --------------------
const client = new ChromaClient({
    url: process.env.CHROMA_URL,       // Если используешь Chroma Cloud
    apiKey: process.env.CHROMA_API_KEY // Ключ Chroma Cloud
});

// -------------------- Создание/получение коллекции --------------------
async function getOrCreateCollection(name) {
    try {
        // пытаемся получить коллекцию
        const existing = await client.getCollection({ name });
        console.log(`Collection "${name}" уже существует ✅`);
        return existing;
    } catch (err) {
        // если не существует, создаём новую
        console.log(`Создаём коллекцию "${name}"...`);
        return await client.createCollection({ name });
    }
}

// -------------------- Чтение файлов и разбивка на чанки --------------------
function walk(dir) {
    const files = [];
    fs.readdirSync(dir, { withFileTypes: true }).forEach((entry) => {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
            files.push(...walk(fullPath));
        } else if (entry.isFile()) {
            files.push(fullPath);
        }
    });
    return files;
}

function chunkText(text, chunkSize = 500) {
    const chunks = [];
    for (let i = 0; i < text.length; i += chunkSize) {
        chunks.push(text.slice(i, i + chunkSize));
    }
    return chunks;
}

// -------------------- Векторизация --------------------
async function embedFile(filePath, collection, embedder) {
    const content = fs.readFileSync(filePath, "utf-8");
    const chunks = chunkText(content);
    for (const [i, chunk] of chunks.entries()) {
        try {
            const embedding = await embedder.embed(chunk);
            await collection.add({
                ids: [`${filePath}-${i}`],
                embeddings: [embedding],
                metadatas: [{ file: filePath, chunk: i }],
                documents: [chunk],
            });
            console.log(`✅ Векторизован chunk ${i} файла ${filePath}`);
        } catch (err) {
            console.error(`Ошибка embedding ${filePath} chunk ${i}:`, err.message);
        }
    }
}

// -------------------- Главная функция --------------------
async function main() {
    console.log("Loading local embedding model...");
    const embedder = new OpenAIEmbeddings({}); // Используем локальные эмбеддинги
    console.log("Model loaded ✅");

    const collection = await getOrCreateCollection(collectionName);

    const files = walk(DATA_DIR);
    console.log(`Найдено файлов для векторизации: ${files.length}`);

    for (const file of files) {
        await embedFile(file, collection, embedder);
    }

    console.log("✅ Vectorization complete");
}

// -------------------- Запуск --------------------
main().catch((err) => console.error("Fatal error:", err));



