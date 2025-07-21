import json
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage, Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# =====================
# 初始化模型和 Embedding
# =====================
llm = ChatOpenAI(
    model_name="deepseek-chat",
    temperature=0.4,
    openai_api_key="sk-f4fc5cb0887b43918ce40ca538e8e06f",
    openai_api_base="https://api.deepseek.com"
)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# =====================
# 加载“小美家电产品库”JSON 数据
# =====================
with open("/data/private/小美家电产品库.json", "r", encoding="utf-8") as f:
    product_data = json.load(f)

docs = [
    Document(
        page_content=item["description"],
        metadata={
            "model": item["model"],
            "category": item["category"],
            "color": item["color"],
            "capacity": item["capacity"],
            "price": item["price"],
            "features": item["features"]
        }
    )
    for item in product_data
]

# =====================
# 构建向量数据库
# =====================
vectorstore = FAISS.from_documents(docs, embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})


# =====================
# Agent 1: 用户需求采集 Agent
# =====================
class RequirementCollector:
    def __init__(self):
        self.collected_tags = []
        self.max_tags = 3
        self.history = [
            SystemMessage(content="""
你是一个善于引导用户描述购买需求的助手。每轮你会根据用户输入输出两个字段：
1. Tag: 用一句话总结用户刚刚的需求，例如“冰箱”、“价格便宜”、“适合三口之家”
2. Next Question: 向用户提出一个新问题，用来获取更多购买偏好。
当你收集到3个tag时，请停止提问，仅输出全部收集的tag列表。
输出格式：
Tag: xxx
Next Question: xxx
""")
        ]

    def step(self, user_input):
        self.history.append(HumanMessage(content=user_input))
        response = llm(self.history)
        self.history.append(AIMessage(content=response.content))

        tag, next_q = self.parse_response(response.content)
        if tag:
            self.collected_tags.append(tag)

        if len(self.collected_tags) >= self.max_tags:
            return "✅ Tag 收集完毕", self.collected_tags, True
        else:
            return f"Tag: {tag}\nNext Question: {next_q}", self.collected_tags, False

    def parse_response(self, text):
        tag = None
        next_q = ""
        for line in text.strip().splitlines():
            if line.startswith("Tag:"):
                tag = line.split("Tag:")[1].strip()
            elif line.startswith("Next Question:"):
                next_q = line.split("Next Question:")[1].strip()
        return tag, next_q


# =====================
# Agent 2: 产品推荐 Agent（根据 tags 检索向量库）
# =====================
class ProductMatcher:
    def __init__(self, retriever):
        self.retriever = retriever

    def match(self, tag_list):
        # 默认将第一个 tag 视为产品类型
        category = tag_list[0] if tag_list else "家电"
        query = ", ".join(tag_list)
        results = self.retriever.get_relevant_documents(f"我想买{category}，要求是：{query}")
        if results:
            doc = results[0]
            return f"🎯 推荐产品：{doc.metadata.get('model')} \n📄 描述：{doc.page_content}"
        else:
            return "❌ 没有找到匹配的产品，请换个方式描述需求。"


# =====================
# 启动问答流程
# =====================
collector = RequirementCollector()
matcher = ProductMatcher(retriever)

print("🤖 你好，我是你的购机助手。我将逐步了解你的需求，然后推荐最合适的产品。\n请问你想买什么？")

while True:
    user_input = input("🧑 用户：")
    agent_reply, tags, done = collector.step(user_input)
    print("🤖 Agent1：", agent_reply)

    if done:
        print("\n✅ 已收集 Tag：", tags)
        result = matcher.match(tags)
        print("\n" + result)
        break
