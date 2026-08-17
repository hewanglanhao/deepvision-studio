import { createHash } from 'node:crypto';
import { createReadStream, createWriteStream } from 'node:fs';
import { copyFile, mkdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { once } from 'node:events';
import { spawn } from 'node:child_process';

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const FRONTEND_ROOT = resolve(SCRIPT_DIR, '..');
const TARGET_ROOT = join(FRONTEND_ROOT, 'public', 'mode-d-assets');
const DEFAULT_SOURCE_BASE = 'https://poloclub.github.io/transformer-explainer';
const TOKENIZER_REVISION = 'bf2c7f02e0b826c60d03af341171bde20893da66';
const TOKENIZER_BASE = `https://huggingface.co/Xenova/gpt2/resolve/${TOKENIZER_REVISION}`;

const MODEL_HASHES = [
  'a20a49894fc5811d5fdc8bd5cec741bc726a1cba',
  'b0f556d6ba0044e922e001f49ba8d83cc7c81f45',
  '2b307e966853fe5067948ae07ff75b4e55f5f8f0',
  'b81e526532a6cf24f08c23537a4cd9694d03c46b',
  'd7372715cb0a3fd5d7332de4e65251119c1a63ef',
  '3736fdbd93d02b8608ebe3c291c2c27acbf782a2',
  '753c1ff91006c17b8ed70091417b36f10c9f206a',
  '6a5ccf79e08c326a8ed580554e09351d9c9725d9',
  '5427a09ee8f9df7cb9d2d542581732cc7b1dfe60',
  '4b990bb223750d0c3f33e8dd335c23f1b7ed838f',
  '669f2e279c1b397a307b68d69d8e6f2626f3464f',
  'a3ddce21ca1bac6af56cd5967c9213b0594ca209',
  '404dcb820320b0ac9aa3a8d122039d13d09f81bb',
  'e2a0d7b0353dcaf827a0f6deab7045cd3dc5c369',
  '9caac2b080d12660671c9db6883eb4bd944595e9',
  '42a1d9acf78ad6f28fa7b0d831bdf7438ce8c4f0',
  '25add4d0d9abb572c04ef1214e5dc253b6e0637d',
  'b89e3c6661294e9021117609f10af66f01c0f002',
  'e6bb5d1dd0d831259e9a6f41cf0ad4a7189eabdf',
  'ccc8500c53ec5b9adfdaa7156525656a3007716c',
  '3142a82032f6ef25fe2d6ec76a2d8585c906dc13',
  '3b4c6f9fe56c49cf7e2138e76dc92dfac8b51ad8',
  'b480067fa6262dbe577e61ad5ab7620872dc478b',
  '9a613e352b1c937e74e3de28eeeb8f116a0e0ae5',
  '9a54f24ce95f774dc404633411dd46974a5e8b8b',
  '0c45fbbebfa28104f321e8b867a234462cd1af9d',
  'f8ef7fa90c3ee178c4cb8f9f44a482920235c78b',
  'c05daf8a21f3529fdc2e31606ddd3e9142fcc5ec',
  'd18ec708bbbb4a9b8cff490dcf187cf9977dd14a',
  '52fb4fad40009dd021957f33e4a564a52169a223',
  'd01557c30c8fac3bf9a61e6678792bade1d69080',
  '4098a0d91ed6054f2d3a7d8eb535371e69da71c4',
  'df72078f0d78456d2d2b8fda12cba5469a047533',
  '27fc201ab6ebaf59542d7da2064ff68c67bbcfe4',
  '922831f0339a4b22a35c4e8b06e30f99e4d52840',
  'd44070045c1925cbd5394c5714ddabf736598746',
  'dc9c1a09ff8adfb41369a13749b1f7bcde77f928',
  '75ff3852ba57be11e4bc562578cbdd8227f91a5f',
  'cbaabff005f4a8460982148881d07db6eb76a32d',
  '7ec76e341f3e631a7395ece4c22cdb1bd4f8c6bd',
  '9dd2beb06f0b6a4cd55f22bbd5d4e65820b8ab10',
  '0883571d7f34213e05bab82d4932986fcd3a78eb',
  '6bd15ca2ee5ad1ca9d1a2847269964c8f7078961',
  '507d578673ae007ef769ae3f445284e657543967',
  '7106a152058ff17184136b772aa7c64d5fdbf6fc',
  'fd244048ddeaebecaea3fb7254a312b0cae0cd19',
  '2bfc68a0598ab1eaaeeb87fb61cf83f052463b6c',
  'c4e91787180a98031cb3013d4d6f3fac76013fc3',
  'cae8461ec203cb6ef65b3e5b9ce122727284f45b',
  'd27ca9ad40ff91b9dc328e9b033589b50589e776',
  'e6a4b2773491b7b9f36e45dac949a3da1b1d9bc1',
  '1746b6dc8278a882ca3f71ce8233e26fc981e98d',
  'ef561626de475718c22a20b69341157ae7a531cd',
  'b83381695deb9aaa24f36e232baf77f688d034dd',
  '9a78f64beee5fe32c97eeeac276c9adf033d81f0',
  'eb351a8d0c4b60abc91e2832646b0692e8bdf3b8',
  '60dfb151235751a76824d3a7dea2d2b5f1766b0b',
  '2fdc6b7f538c6701af6761d8edbeeae8f76ecb8d',
  'b36d23c7893bf205dec85218cefa00a818a9b81f',
  '80eecaae6a7427cbe6a24ac1a3ffbaad939a3c28',
  'c32b7300ddfd79707697cdc0ac80d87037a6b337',
  '957b721eae1d89e0ce7fc3098532c406b77cf9c8',
  'd8780552151456f43b8f40fa2562f794540baa16'
];

const MODEL_PARTS = MODEL_HASHES.map((gitSha1, index) => ({
  name: `gpt2.onnx.part${index}`,
  size: index === MODEL_HASHES.length - 1 ? 6_545_544 : 10_485_760,
  gitSha1
}));

const RUNTIME_FILES = [
  {
    source: join(FRONTEND_ROOT, 'node_modules', 'onnxruntime-web', 'dist', 'ort-wasm-simd-threaded.jsep.mjs'),
    targets: [
      join(TARGET_ROOT, 'vendor', 'onnxruntime', 'ort-wasm-simd-threaded.jsep.mjs'),
      join(TARGET_ROOT, 'vendor', 'onnxruntime', 'ort-wasm-simd-threaded.jsep.js')
    ]
  },
  {
    source: join(FRONTEND_ROOT, 'node_modules', 'onnxruntime-web', 'dist', 'ort-wasm-simd-threaded.jsep.wasm'),
    targets: [join(TARGET_ROOT, 'vendor', 'onnxruntime', 'ort-wasm-simd-threaded.jsep.wasm')]
  },
  {
    source: join(FRONTEND_ROOT, 'node_modules', '@xenova', 'transformers', 'dist', 'transformers.min.js'),
    targets: [join(TARGET_ROOT, 'vendor', 'transformers', 'transformers.min.js')]
  }
];

const TOKENIZER_FILES = [
  {
    name: 'config.json',
    size: 884,
    sha256: 'c9e2a8cc16fced63fa05e353a330ec236bf52d16f01d27c18ee50f39849a39a5'
  },
  {
    name: 'tokenizer.json',
    size: 2_107_653,
    sha256: 'cda20b8ca044949aa07ac4078420c80d1a57139d5f9f33700e46fb2d891e7c66'
  },
  {
    name: 'tokenizer_config.json',
    size: 234,
    sha256: '551e26ec611d8d0c8edc3ef72e518a38418cb71f40de1347dd486a595e1557d7'
  }
];

function parseOptions(argv) {
  const options = {
    check: false,
    force: false,
    concurrency: 4,
    sourceBase: DEFAULT_SOURCE_BASE
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--check') options.check = true;
    else if (arg === '--force') options.force = true;
    else if (arg === '--concurrency') options.concurrency = Number(argv[++index]);
    else if (arg === '--source-base') options.sourceBase = argv[++index];
    else if (arg === '--help') {
      console.log(`Usage: node scripts/setup-mode-d-assets.mjs [options]\n\n` +
        `  --check              Verify local assets without downloading\n` +
        `  --force              Download model and tokenizer files again\n` +
        `  --concurrency <n>    Parallel model downloads (default: 4)\n` +
        `  --source-base <url>  Override the Transformer Explainer asset mirror`);
      process.exit(0);
    } else {
      throw new Error(`Unknown option: ${arg}`);
    }
  }

  if (!Number.isInteger(options.concurrency) || options.concurrency < 1 || options.concurrency > 8) {
    throw new Error('--concurrency must be an integer from 1 to 8.');
  }
  options.sourceBase = options.sourceBase.replace(/\/$/, '');
  return options;
}

async function fileSize(path) {
  try {
    return (await stat(path)).size;
  } catch {
    return -1;
  }
}

async function gitBlobSha1(path, size) {
  const hash = createHash('sha1');
  hash.update(Buffer.from(`blob ${size}\0`));
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}

async function sha256(path) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}

async function validModelPart(path, part) {
  if ((await fileSize(path)) !== part.size) return false;
  return (await gitBlobSha1(path, part.size)) === part.gitSha1;
}

async function download(
  url,
  target,
  { expectedSize, expectedGitSha1, expectedSha256, force = false } = {}
) {
  if (!force) {
    const currentSize = await fileSize(target);
    if (expectedSize == null ? currentSize > 0 : currentSize === expectedSize) {
      const gitHashMatches = !expectedGitSha1 || (await gitBlobSha1(target, expectedSize)) === expectedGitSha1;
      const sha256Matches = !expectedSha256 || (await sha256(target)) === expectedSha256;
      if (gitHashMatches && sha256Matches) return false;
    }
  }

  await mkdir(dirname(target), { recursive: true });
  const temp = `${target}.download`;
  await rm(temp, { force: true });
  try {
    try {
      await downloadWithCurl(url, temp);
    } catch (curlError) {
      await rm(temp, { force: true });
      console.warn(`curl is unavailable or failed for ${url}; retrying with Node fetch.`);
      try {
        await downloadWithFetch(url, temp);
      } catch (fetchError) {
        const fetchMessage = fetchError instanceof Error ? fetchError.message : String(fetchError);
        const curlMessage = curlError instanceof Error ? curlError.message : String(curlError);
        throw new Error(`${url}\nNode fetch: ${fetchMessage}\ncurl: ${curlMessage}`);
      }
    }

    const received = await fileSize(temp);
    if (expectedSize != null && received !== expectedSize) {
      throw new Error(`Size mismatch for ${url}: expected ${expectedSize}, received ${received}`);
    }
    if (expectedGitSha1 && (await gitBlobSha1(temp, expectedSize)) !== expectedGitSha1) {
      throw new Error(`Integrity mismatch for ${url}; the upstream asset version changed.`);
    }
    if (expectedSha256 && (await sha256(temp)) !== expectedSha256) {
      throw new Error(`SHA-256 mismatch for ${url}; the tokenizer asset version changed.`);
    }

    await rm(target, { force: true });
    await rename(temp, target);
    return true;
  } catch (error) {
    await rm(temp, { force: true });
    throw error;
  }
}

async function downloadWithFetch(url, target) {
  const response = await fetch(url, {
    redirect: 'follow',
    signal: AbortSignal.timeout(300_000),
    headers: { 'user-agent': 'DeepVision-Studio-Mode-D-Setup/1.0' }
  });
  if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);

  const output = createWriteStream(target, { flags: 'w' });
  try {
    for await (const chunk of response.body) {
      if (!output.write(Buffer.from(chunk))) await once(output, 'drain');
    }
    output.end();
    await once(output, 'finish');
  } catch (error) {
    output.destroy();
    throw error;
  }
}

async function downloadWithCurl(url, target) {
  await new Promise((resolvePromise, rejectPromise) => {
    const child = spawn(
      'curl',
      [
        '--fail', '--location', '--silent', '--show-error',
        '--retry', '3', '--retry-delay', '1',
        '--connect-timeout', '15', '--max-time', '300',
        '--output', target, url
      ],
      { stdio: 'inherit', windowsHide: true }
    );
    child.once('error', rejectPromise);
    child.once('exit', code => {
      if (code === 0) resolvePromise();
      else rejectPromise(new Error(`curl exited with code ${code}`));
    });
  });
}

async function copyRuntimeFiles() {
  for (const item of RUNTIME_FILES) {
    if ((await fileSize(item.source)) <= 0) {
      throw new Error(`Missing ${item.source}. Run "npm ci" in frontend first.`);
    }
    for (const target of item.targets) {
      await mkdir(dirname(target), { recursive: true });
      await copyFile(item.source, target);
    }
  }
}

async function downloadModelParts(options) {
  console.log(`Mode D model: ${MODEL_PARTS.length} chunks, 626.2 MiB total.`);
  let cursor = 0;
  let downloaded = 0;
  let reused = 0;

  async function worker() {
    while (true) {
      const index = cursor++;
      if (index >= MODEL_PARTS.length) return;
      const part = MODEL_PARTS[index];
      const target = join(TARGET_ROOT, 'model-v2', part.name);
      const changed = await download(
        `${options.sourceBase}/model-v2/${part.name}`,
        target,
        { expectedSize: part.size, expectedGitSha1: part.gitSha1, force: options.force }
      );
      changed ? downloaded++ : reused++;
      console.log(`[${downloaded + reused}/${MODEL_PARTS.length}] ${changed ? 'downloaded' : 'verified'} ${part.name}`);
    }
  }

  await Promise.all(Array.from({ length: options.concurrency }, () => worker()));
}

async function downloadTokenizer(force) {
  for (const file of TOKENIZER_FILES) {
    const target = join(TARGET_ROOT, 'models', 'Xenova', 'gpt2', file.name);
    const changed = await download(`${TOKENIZER_BASE}/${file.name}`, target, {
      expectedSize: file.size,
      expectedSha256: file.sha256,
      force
    });
    console.log(`${changed ? 'Downloaded' : 'Verified'} tokenizer/${file.name}`);
  }
}

async function readPackageVersion(path) {
  const contents = JSON.parse(await readFile(path, 'utf8'));
  return contents.version;
}

async function writeManifest(sourceBase) {
  const manifest = {
    schemaVersion: 1,
    installedAt: new Date().toISOString(),
    source: {
      transformerExplainer: sourceBase,
      tokenizer: `Xenova/gpt2@${TOKENIZER_REVISION}`
    },
    model: {
      chunks: MODEL_PARTS.length,
      totalBytes: MODEL_PARTS.reduce((sum, part) => sum + part.size, 0)
    },
    runtime: {
      onnxruntimeWeb: await readPackageVersion(join(FRONTEND_ROOT, 'node_modules', 'onnxruntime-web', 'package.json')),
      transformersJs: await readPackageVersion(join(FRONTEND_ROOT, 'node_modules', '@xenova', 'transformers', 'package.json'))
    }
  };
  await writeFile(join(TARGET_ROOT, 'asset-manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
}

async function checkAssets() {
  const problems = [];
  for (let index = 0; index < MODEL_PARTS.length; index += 1) {
    const part = MODEL_PARTS[index];
    const path = join(TARGET_ROOT, 'model-v2', part.name);
    if (!(await validModelPart(path, part))) problems.push(`invalid or missing model-v2/${part.name}`);
    if ((index + 1) % 10 === 0 || index === MODEL_PARTS.length - 1) {
      console.log(`Checked ${index + 1}/${MODEL_PARTS.length} model chunks.`);
    }
  }

  for (const item of RUNTIME_FILES.flatMap(item => item.targets)) {
    if ((await fileSize(item)) <= 0) problems.push(`missing ${item.slice(TARGET_ROOT.length + 1)}`);
  }
  for (const file of TOKENIZER_FILES) {
    const path = join(TARGET_ROOT, 'models', 'Xenova', 'gpt2', file.name);
    if ((await fileSize(path)) !== file.size || (await sha256(path)) !== file.sha256) {
      problems.push(`invalid or missing models/Xenova/gpt2/${file.name}`);
    }
  }

  if (problems.length) {
    throw new Error(`Mode D asset check failed:\n- ${problems.join('\n- ')}`);
  }
  console.log('Mode D assets are complete and verified.');
}

async function main() {
  const options = parseOptions(process.argv.slice(2));
  if (options.check) {
    await checkAssets();
    return;
  }

  await mkdir(TARGET_ROOT, { recursive: true });
  await copyRuntimeFiles();
  await downloadTokenizer(options.force);
  await downloadModelParts(options);
  await writeManifest(options.sourceBase);
  await checkAssets();
}

main().catch(error => {
  console.error(`\nMode D setup failed: ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
});
