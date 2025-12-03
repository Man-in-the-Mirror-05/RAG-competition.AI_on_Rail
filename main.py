from src.rag_pipeline import run_pipeline


if __name__ == "__main__":
    outputs = run_pipeline()
    answer_path = outputs.get("answers")
    performance_path = outputs.get("performance")
    print(f"所有问题的回答已保存到：{answer_path}")
    print(f"性能统计信息已保存到：{performance_path}")
