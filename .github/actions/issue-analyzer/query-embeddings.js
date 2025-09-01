import { pipeline } from "@xenova/transformers";
import { CloudClient } from "chromadb";

// ====== CONFIG ======
const CHROMA_API_KEY = process.env.CHROMA_API_KEY || '';
const CHROMA_TENANT = process.env.CHROMA_TENANT || '';
const CHROMA_DATABASE = process.env.CHROMA_DATABASE || '';
const COLLECTION_NAME = "FullProjectCollection";
const MAX_RETRIES = 3;
const MAX_EMBED_DIM = 3072; // лимит Chroma Starter

// ====== UTILS ======
async function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function withRetry(fn, retries = MAX_RETRIES, delay = 1000) {
    let lastError;
    for (let i = 0; i < retries; i++) {
        try {
            return await fn();
        } catch (err) {
            lastError = err;
            console.warn(`Attempt ${i + 1} failed:`, err.message);
            await sleep(delay * (i + 1));
        }
    }
    throw lastError;
}

// Уменьшаем размерность эмбеддинга до лимита
function resizeEmbedding(embedding, maxDim = MAX_EMBED_DIM) {
    if (embedding.length <= maxDim) return embedding;
    const factor = embedding.length / maxDim;
    const resized = [];
    for (let i = 0; i < maxDim; i++) {
        const idx = Math.floor(i * factor);
        resized.push(embedding[idx]);
    }
    return resized;
}

// ====== MAIN ======
async function main() {
    try {
        console.log("🚀 Starting issue analysis...");

        if (!CHROMA_API_KEY || !CHROMA_TENANT || !CHROMA_DATABASE) {
            console.error("❌ Missing Chroma credentials");
            process.exit(1);
        }

        // 1️⃣ Vectorize issue
        console.log("🔍 Vectorizing issue text...");
        const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
            quantized: true,
            device: "cpu",
            pooling: "mean",
            normalize: true
        });

        const ISSUE_BODY = process.env.ISSUE_BODY || '';
        if (!ISSUE_BODY) {
            console.error("❌ ISSUE_BODY is empty");
            process.exit(1);
        }

        const embeddingOutput = await embedder(ISSUE_BODY);
        let issueEmbedding;

        if (embeddingOutput && 'data' in embeddingOutput) {
            issueEmbedding = Array.from(embeddingOutput.data);
        } else if (Array.isArray(embeddingOutput)) {
            issueEmbedding = embeddingOutput.flat(Infinity);
        } else {
            console.error("❌ Unexpected embedding output format");
            process.exit(1);
        }

        // Resize до лимита Chroma
        issueEmbedding = resizeEmbedding(issueEmbedding);

        console.log("✅ Issue text embedded (length:", issueEmbedding.length, ")");

        // 2️⃣ Connect to Chroma
        console.log("🔌 Connecting to Chroma...");
        const client = new CloudClient({
            apiKey: CHROMA_API_KEY,
            tenant: CHROMA_TENANT,
            database: CHROMA_DATABASE
        });

        const collection = await withRetry(async () => {
            const col = await client.getCollection({ name: COLLECTION_NAME });
            console.log("✅ Connected to collection:", col.name);
            return col;
        });

        // 3️⃣ Query relevant chunks
        console.log("🔎 Searching for relevant code...");
        const results = await withRetry(async () => {
            const res = await collection.query({
                query_embeddings: [issueEmbedding],
                n_results: 5
            });

            if (!res || !res.documents || res.documents.length === 0) {
                throw new Error("No results from Chroma");
            }
            return res;
        });

        console.log("✅ Found relevant chunks:");
        results[0].documents.forEach((doc, idx) => {
            const meta = results[0].metadatas[idx];
            console.log(`- [${meta.file} | chunk ${meta.chunkId}]`);
        });

    } catch (err) {
        console.error("❌ Fatal error:", err);
        process.exit(1);
    }
}

main();
