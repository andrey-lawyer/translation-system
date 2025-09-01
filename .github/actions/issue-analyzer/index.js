import { CloudClient } from "chromadb";
import { pipeline } from "@xenova/transformers";
import { Octokit } from "@octokit/rest";

// ====== ENVIRONMENT VALIDATION ======
const requiredEnvVars = [
    'CHROMA_API_KEY',
    'CHROMA_TENANT',
    'CHROMA_DATABASE',
    'ISSUE_BODY',
    'ISSUE_NUMBER',
    'GH_PAT',
    'GITHUB_REPOSITORY'
];

for (const envVar of requiredEnvVars) {
    if (!process.env[envVar]) {
        console.error(`❌ Missing required environment variable: ${envVar}`);
        process.exit(1);
    }
}

const {
    CHROMA_API_KEY,
    CHROMA_TENANT,
    CHROMA_DATABASE,
    ISSUE_BODY,
    ISSUE_NUMBER,
    GH_PAT,
    GITHUB_REPOSITORY
} = process.env;

const COLLECTION_NAME = "FullProjectCollection";
const MAX_RETRIES = 3;
const RETRY_DELAY = 2000; // ms
const MAX_EMBED_DIM = 3072; // Chroma Starter limit

// ====== UTILS ======
async function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function withRetry(fn, retries = MAX_RETRIES, delay = RETRY_DELAY) {
    let lastError;
    for (let i = 0; i < retries; i++) {
        try {
            return await fn();
        } catch (err) {
            lastError = err;
            console.warn(`Attempt ${i + 1} failed:`, err.message);
            if (i < retries - 1) await sleep(delay * (i + 1));
        }
    }
    throw lastError;
}

// Resize embedding до лимита
function resizeEmbedding(embedding, maxDim = MAX_EMBED_DIM) {
    if (embedding.length <= maxDim) return embedding;
    const factor = embedding.length / maxDim;
    const resized = [];
    for (let i = 0; i < maxDim; i++) {
        resized.push(embedding[Math.floor(i * factor)]);
    }
    return resized;
}

// ====== MAIN ======
async function main() {
    try {
        console.log('🚀 Starting issue analysis...');

        // 1️⃣ Vectorize issue
        console.log('🔍 Vectorizing issue text...');
        const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
            quantized: true,
            device: "cpu",
            pooling: "mean",
            normalize: true
        });

        let embeddingOutput = await withRetry(() => embedder(ISSUE_BODY));
        let issueEmbedding;

        if (embeddingOutput && 'data' in embeddingOutput) {
            issueEmbedding = Array.from(embeddingOutput.data);
        } else if (Array.isArray(embeddingOutput)) {
            // flatten + mean
            const flat = embeddingOutput.flat(1);
            const dim = flat[0]?.length || flat.length;
            issueEmbedding = new Array(dim).fill(0);
            flat.forEach(vec => {
                vec.forEach((val, idx) => {
                    issueEmbedding[idx] += val / flat.length;
                });
            });
        } else {
            throw new Error('Unexpected embedding output format');
        }

        issueEmbedding = resizeEmbedding(issueEmbedding, MAX_EMBED_DIM);
        console.log("✅ Issue embedding ready (length:", issueEmbedding.length, ")");

        // 2️⃣ Connect to Chroma
        console.log('🔌 Connecting to Chroma...');
        const client = new CloudClient({
            apiKey: CHROMA_API_KEY,
            tenant: CHROMA_TENANT,
            database: CHROMA_DATABASE
        });

        const collection = await withRetry(async () => {
            const col = await client.getCollection({ name: COLLECTION_NAME });
            console.log(`✅ Connected to collection: ${col.name}`);
            return col;
        });

        // 3️⃣ Query relevant chunks
        console.log('🔎 Searching for relevant code...');
        const results = await withRetry(async () => {
            const res = await collection.query({
                queryEmbeddings: [issueEmbedding],
                nResults: 5,
                include: ["metadatas", "distances"]
            });
            if (!res || !res.metadatas || res.metadatas.length === 0) {
                throw new Error('No results from Chroma');
            }
            return res;
        });

        const metadatas = results.metadatas[0];
        const distances = results.distances?.[0];

        console.log('✅ Found relevant chunks:');
        for (let i = 0; i < metadatas.length; i++) {
            const meta = metadatas[i];
            const file = meta?.file ?? '[unknown]';
            const chunkId = meta?.chunkId ?? 0;
            const distance = distances?.[i]?.toFixed(2) ?? 'n/a';
            console.log(`- [${file} | chunk ${chunkId} | distance: ${distance}] -> [text not available]`);
        }

        // 4️⃣ Prepare patch
        console.log('🤖 Preparing patch...');
        // берём относительный путь относительно корня репозитория
        let patchFile = metadatas[0]?.file;
        if (!patchFile) throw new Error("No valid file found for patch");

        // убираем абсолютные пути (например, /home/runner/work/.../repo-name/)
        const repoRoot = process.cwd();
        patchFile = path.relative(repoRoot, patchFile).replace(/^\/+/, "");
        console.log(`📂 Normalized patch file path: ${patchFile}`);
        const patchContent = `// Auto-generated fix for issue #${ISSUE_NUMBER}\n// Placeholder content\n`;




        // 5️⃣ GitHub: branch, commit, PR
        console.log('🌿 Connecting to GitHub...');
        const [owner, repo] = GITHUB_REPOSITORY.split('/');
        const octokit = new Octokit({ auth: GH_PAT });

        const branchName = `issue-${ISSUE_NUMBER}`;
        console.log(`Creating branch: ${branchName}...`);
        const { data: mainRef } = await withRetry(() =>
            octokit.git.getRef({ owner, repo, ref: 'heads/main' })
        );

        await withRetry(() =>
            octokit.git.createRef({
                owner,
                repo,
                ref: `refs/heads/${branchName}`,
                sha: mainRef.object.sha
            })
        );
        console.log(`✅ Branch ${branchName} created`);

        // Commit file
        let fileSha;
        try {
            const { data: fileData } = await octokit.repos.getContent({ owner, repo, path: patchFile });
            fileSha = fileData.sha;
        } catch (err) {
            if (err.status !== 404) throw err;
        }

        console.log(`Committing changes to ${patchFile}...`);
        await withRetry(() =>
            octokit.repos.createOrUpdateFileContents({
                owner,
                repo,
                path: patchFile,
                message: `Fix for issue #${ISSUE_NUMBER}`,
                content: Buffer.from(patchContent).toString('base64'),
                sha: fileSha,
                branch: branchName
            })
        );
        console.log(`✅ Changes committed to ${patchFile}`);

        // Create PR
        console.log('Creating pull request...');
        const pr = await withRetry(() =>
            octokit.pulls.create({
                owner,
                repo,
                title: `Fix for issue #${ISSUE_NUMBER}`,
                head: branchName,
                base: 'main',
                body: `Automated fix for issue #${ISSUE_NUMBER}`
            })
        );
        console.log(`✅ Pull request created: ${pr.data.html_url}`);

        console.log('✅ All done!');

    } catch (err) {
        console.error('❌ Fatal error:', err);
        process.exit(1);
    }
}

main();
