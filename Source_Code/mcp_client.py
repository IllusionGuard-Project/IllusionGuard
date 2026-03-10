# -*- coding: utf-8 -*-
import os, csv, json, argparse, asyncio
from contextlib import AsyncExitStack
from typing import List
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# # 你的固定接口（宏工具）名 —— 与 server.py 保持一致
# LABEL_TO_TOOL = {
#     "AI_unsafe_illusion": "tp_ai_lowfreq_read",       # 低频读取
#     "HighLowFreq":        "tp_highlow_dualpass",      # 高低频双图
#     "DoubleLayer":        "tp_doublelayer_phantom",   # 幻影坦克两层
#     "striped_words":      "tp_striped_notch",
#     "DirectionIllusion":  "tp_direction_autocompress",
#     "IllusoryOutline":    "tp_illusory_outline",
#     "normal_unsafe_word": "tp_normal_unsafe",
# }
# 你的固定接口（宏工具）名 —— 与 server.py 保持一致
LABEL_TO_TOOL = {
    "LFF": "tp_ai_lowfreq_read",       # 低频读取
    "HLH":        "tp_highlow_dualpass",      # 高低频双图
    "DLT":        "tp_doublelayer_phantom",   # 幻影坦克两层
    "SO":      "tp_striped_notch",
    "DC":  "tp_direction_autocompress",
    "IO":    "tp_illusory_outline",
    "normal_word": "tp_normal_unsafe",
    "no_illusion": "tp_ai_lowfreq_read",
}
async def call_mcp_tool(session: ClientSession, tool: str, args: dict) -> dict:
    res = await session.call_tool(tool, args)
    # FastMCP -> result.content[0].text
    txt = res.content[0].text if getattr(res, "content", None) else ""
    return json.loads(txt)

def read_csv_rows(p: str) -> List[dict]:
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def write_csv_rows(p: str, rows: List[dict]):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        keys = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

async def main(args):
    # 连接 MCP server
    exit_stack = AsyncExitStack()
    params = StdioServerParameters(command="python", args=[args.server])
    stdio_t = await exit_stack.enter_async_context(stdio_client(params))
    stdio, write = stdio_t
    session = await exit_stack.enter_async_context(ClientSession(stdio, write))
    await session.initialize()

    # VLM 客户端（本地 vLLM 的 OpenAI 兼容端点）
    openai = OpenAI(api_key=args.api_key, base_url=args.base_url)

    rows = read_csv_rows(args.in_csv)
    out_rows = []

    for row in rows:
        img = row.get("image_path") or row.get("image") or ""
        label = row.get("pred_label") or row.get("label") or  row.get("category") or ""
        if not img or not os.path.exists(img):
            row["recognized_text"] = ""
            row["error"] = f"missing image: {img}"
            out_rows.append(row); continue

        tool = LABEL_TO_TOOL.get(label)
        if not tool:
            row["recognized_text"] = ""
            row["error"] = f"unknown label: {label}"
            out_rows.append(row); continue

        # 调该类固定接口（宏工具）
        try:
            plan = await call_mcp_tool(session, tool, {"src": img, "outdir": args.outdir})
        except Exception as e:
            row["recognized_text"] = ""
            row["error"] = f"mcp failed: {e}"
            out_rows.append(row); continue
        
        # outs = plan.get("outs", [])
        # vlm_prompt = plan.get("vlm_prompt", "请识别图中文字，逐字输出。")
        # answers = []

        # for p in outs:
        #     url = f"file://{os.path.abspath(p)}"
        #     resp = openai.chat.completions.create(
        #         model=args.model,
        #         messages=[{
        #             "role":"user",
        #             "content":[
        #                 {"type":"text","text": vlm_prompt},
        #                 {"type":"image_url","image_url":{"url": url}}
        #             ]
        #         }],
        #         max_tokens=256, temperature=0
        #     )
        #     ans = (resp.choices[0].message.content or "").strip()
        #     answers.append(ans)

        # row["recognized_text"] = "\n".join(answers)

        # 1) 取 outs + prompts（prompts 可选）
        outs = plan.get("outs", [])
        prompts_list = plan.get("prompts")  # 可能不存在
        COT_OCR_DEFAULT = (
                "You will read a polished image that exposes a hidden English word.\n"
                "Think through the reading in your mind, but output only the final word.\n\n"
                "Strict output rules:\n"
                "- Output exactly ONE lowercase English word (a–z, 2–16 letters).\n"
                "- No explanations, no punctuation, no quotes, no tags.\n"
                "- If any letter is uncertain, replace it with '?'.\n"
                "Now output only the word."
            )
        vlm_prompt_default = plan.get("vlm_prompt", COT_OCR_DEFAULT)

        answers = []

        # 2) 逐图识别：若有 per-image prompt 用之，否则用统一 prompt
        for i, p in enumerate(outs):
            url = f"file://{os.path.abspath(p)}"
            this_prompt = (
                prompts_list[i] if (isinstance(prompts_list, list) and i < len(prompts_list))
                else vlm_prompt_default
            )

            resp = openai.chat.completions.create(
                model=args.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": this_prompt},
                        {"type": "image_url", "image_url": {"url": url}}
                    ]
                }],
                max_tokens=200,
                temperature=0
            )
            ans = (resp.choices[0].message.content or "").strip()
            answers.append(ans)

        row["recognized_text"] = "\n".join(answers)

        out_rows.append(row)

    write_csv_rows(args.out_csv, out_rows)
    print(f"[OK] wrote recognition CSV -> {args.out_csv}")

    await exit_stack.aclose()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv",  required=True, help="阶段1产出的预测CSV（含 image_path, pred_label）")
    ap.add_argument("--out_csv", required=True, help="写入带识别结果的CSV")
    ap.add_argument("--server",  required=True, help="MCP server.py 路径（含固定宏工具）")
    ap.add_argument("--outdir",  default="", help="中间结果输出目录（为空则跟随原图目录）")
    # VLM / vLLM
    ap.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api_key",  default="EMPTY")
    ap.add_argument("--model",    default="/data1/home/pankun/Visual-RFT/Qwen2-VL-7B-Instruct-CLEAN")
    args = ap.parse_args()
    asyncio.run(main(args))
