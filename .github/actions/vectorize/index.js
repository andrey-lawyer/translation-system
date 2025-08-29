import fs from "fs";
import path from "path";
import fetch from "node-fetch";

const CHUNK_SIZE = 500;
const DELAY_MS = 200;

// Здесь мы всё ещё используем Chroma для хранения
import { CloudClient } from "chromadb";
const chroma = new CloudClient({
    apiKey: process.env.CHROMA_API_KEY,
    tenant: process.env.CHROMA_TENANT,
    database: process.env.CHROMA_DATABASE
});

// Hugging Face embedding
async function getEmbedding(text) {
    const res = await fetch("https://api-inference.huggingface.co/embeddings/sentence-transformers/all-MiniLM-L6-v2", {
        method: "POST",
        headers: {
            "Authorization": `Bearer ${process.env.HF_API_KEY}`,
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ inputs: text })
    });

    if (!res.ok) {
        const errText = await res.text();
        throw new Error(`HF API error: ${res.status} ${errText}`);
    }

    const data = await res.json();
    return data.embedding;
}

async function sleep(ms) {
    return new Promise(res => setTimeout(res, ms));
}

async function embedFile(filePath) {
    const stats = fs.statSync(filePath);
    if (stats.size > 200_000) return;

    const content = fs.readFileSync(filePath, "utf-8");
    if (!content.trim()) return;

    const lines = content.split("\n");

    for (let i = 0; i < lines.length; i += CHUNK_SIZE) {
        const chunk = lines.slice(i, i + CHUNK_SIZE).join("\n");

        try {
            const embedding = await getEmbedding(chunk);

            const collection = await chroma.getCollection({ name: "translation-system" })
                .catch(async () => await chroma.createCollection({ name: "translation-system" }));

            await collection.add({
                ids: [`${filePath}-${i}`],
                embeddings: [embedding],
                documents: [chunk],
                metadatas: [{ file: filePath, start_line: i }]
            });

            console.log(`Indexed chunk ${i} of ${filePath}`);
            await sleep(DELAY_MS);
        } catch (err) {
            console.error(`Error embedding ${filePath} chunk ${i}:`, err.message || err);
        }
    }
}

async function walk(dir) {
    const files = fs.readdirSync(dir, { withFileTypes: true });
    for (const file of files) {
        const filePath = path.join(dir, file.name);

        if (file.isDirectory()) {
            if ([".git", "node_modules", "dist", "target"].includes(file.name)) continue;
            await walk(filePath);
        } else if (/\.(go|java|ts|tsx|js|md)$/i.test(file.name)) {
            await embedFile(filePath);
        }
    }
}

(async () => {
    console.log("Starting vectorization (Hugging Face)...");
    await walk(".");
    console.log("✅ Vectorization complete");
})();
