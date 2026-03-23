test 目录用于对 graph 做整体验证和离线评测，当前基于 100 条旅行问题样本运行。

目录说明
1. run_graph_perf.py
   用于性能压测，统计延迟、吞吐、状态分布等指标。
2. run_graph_eval.py
   用于效果评测，除了性能指标，还会结合 verification 和 expected_route 统计质量指标。
3. travel_queries_100.jsonl
   包含 100 条测试问题，每条数据带有 expected_route 等字段，可用于评估路由是否正确。
4. results
   存放测试输出结果，包含 summary 和 detail 两类 JSON 文件。

性能压测指标
1. success_rate
   100 条请求里成功返回的比例。
2. avg_latency_sec / p50 / p90 / p95 / max
   响应延迟统计。
3. throughput_qps
   吞吐量。
4. status_distribution
   `completed`、`needs_confirmation`、`error` 等状态分布。
5. route_distribution
   各 agent 路由分布。
6. answer_source_distribution
   不同回答来源的分布情况。

效果评测指标
1. route_accuracy
   graph 的实际路由与 `expected_route` 的一致率。
2. completed_rate
   最终状态为 `completed` 的比例。
3. needs_confirmation_rate
   需要用户确认的比例。
4. verification_complete_rate
   `final_summary.verification.is_complete` 为 `true` 的比例。
5. avg_latency_sec / p50 / p90 / p95 / max
   响应延迟统计。
6. throughput_qps
   吞吐量。
7. route_distribution
   各类问题最终落到哪个 agent 的分布。
8. answer_source_distribution
   回答来源分布，例如 `ticket_mcp`、`roadmap_mcp`、`rag_service`、`general_llm`。

运行方式
1. 运行性能压测
   python test/run_graph_perf.py

2. 运行效果评测
   python test/run_graph_eval.py

3. 指定并发度
   python test/run_graph_perf.py --concurrency 5
   python test/run_graph_eval.py --concurrency 5

4. 指定 user_id
   python test/run_graph_perf.py --user-id 2
   python test/run_graph_eval.py --user-id 2

结果文件说明
1. detail 文件
   保存每条问题的执行结果，便于排查 `answer_source`、`verification`、错误信息等细节。
2. summary 文件
   保存整体统计结果，适合直接看模型表现、路由效果和总体指标。

说明
1. 这套测试主要用于从整体验证 graph 的行为，包括 LLM、RAG、MCP 路由的综合表现。
2. 如果出现异常，优先查看 detail 文件里的 error 字段，再结合 summary 中的 failure_count 和 success_rate 分析。
