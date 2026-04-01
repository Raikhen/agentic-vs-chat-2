export const dynamic = "force-dynamic";

import fs from "fs";
import path from "path";
import { CompareView } from "./compare-view";

interface RawSample {
  sample_id: number;
  adversarial_prompt: string;
  plan_prompt: string;
  model_used: string;
  risk_domain: string;
  risk_subdomain: string;
}

export interface Sample {
  sample_id: number;
  adversarial_prompt: string;
  plans: Record<string, string>; // model -> plan_prompt
  risk_domain: string;
  risk_subdomain: string;
}

export default function Page() {
  // Try cwd-relative first (works when run from tools/prompt-compare),
  // fall back to scanning up for the data directory
  let filePath = path.join(process.cwd(), "..", "..", "data", "plan_attack", "plans.jsonl");
  if (!fs.existsSync(filePath)) {
    // If cwd is the repo root (e.g. launched from there)
    filePath = path.join(process.cwd(), "data", "plan_attack", "plans.jsonl");
  }
  const raw = fs.readFileSync(filePath, "utf-8");
  const rows: RawSample[] = raw
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line));

  // Group by sample_id, collecting plans from each model
  const byId = new Map<number, Sample>();
  const modelSet = new Set<string>();

  for (const row of rows) {
    modelSet.add(row.model_used);
    if (!byId.has(row.sample_id)) {
      byId.set(row.sample_id, {
        sample_id: row.sample_id,
        adversarial_prompt: row.adversarial_prompt,
        plans: {},
        risk_domain: row.risk_domain,
        risk_subdomain: row.risk_subdomain,
      });
    }
    byId.get(row.sample_id)!.plans[row.model_used] = row.plan_prompt;
  }

  const samples = Array.from(byId.values()).sort((a, b) => a.sample_id - b.sample_id);
  const models = Array.from(modelSet).sort();

  return <CompareView samples={samples} models={models} />;
}
