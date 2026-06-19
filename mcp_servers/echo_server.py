"""最小 MCP echo server，用于端到端测试。"""
import json
import sys


def main() -> None:
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        msg = json.loads(line.strip())
        req_id = msg.get("id")

        if msg.get("method") == "initialize":
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "echo-server", "version": "1.0"},
                },
            }
        elif msg.get("method") == "tools/list":
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echo back the input message",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "message": {
                                        "type": "string",
                                        "description": "The message to echo",
                                    }
                                },
                                "required": ["message"],
                            },
                        }
                    ]
                },
            }
        elif msg.get("method") == "tools/call":
            args = msg.get("params", {}).get("arguments", {})
            echo_text = args.get("message", "no message")
            resp = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"ECHO: {echo_text}"}],
                },
            }
        else:
            # 未知方法或无 id——通知类消息不应响应
            if req_id is None:
                continue
            resp = {"jsonrpc": "2.0", "id": req_id, "result": {}}

        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()