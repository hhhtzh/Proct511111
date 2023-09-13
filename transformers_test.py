import torch
from transformers import AutoTokenizer, AutoModel

# 加载预训练的BERT模型和分词器（你可以选择其他模型）
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


# 将数学表达式转换为句向量
def expression_to_sentence_vector(expression):
    # 分词
    tokens = tokenizer.encode(expression, add_special_tokens=True)

    # 将分词转换为PyTorch张量
    input_ids = torch.tensor(tokens).unsqueeze(0)

    # 获取句向量
    with torch.no_grad():
        outputs = model(input_ids)
        sentence_vector = outputs.last_hidden_state.mean(dim=1).squeeze()

    return sentence_vector


# 示例数学表达式列表
expressions = ["x+y", "3*(a-b)", "sqrt(x^2+y^2)"]

# 生成句向量
for expression in expressions:
    vector = expression_to_sentence_vector(expression)
    print(f"Expression: {expression}")
    print(f"Sentence Vector: {vector}")
    print()
