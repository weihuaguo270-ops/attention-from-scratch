"""
数据加载 — TinyStories 文本数据处理

支持两种模式：
  1. 嵌入示例数据（无外部依赖，即刻运行）
  2. 下载 TinyStories 数据集（需网络）
"""
import os
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader


# 嵌入的 TinyStories 示例（无网络时使用）
SAMPLE_STORIES = [
    "Once upon a time there was a little girl named Lily She loved to play in the forest near her house One day she found a small rabbit hiding under a bush The rabbit was scared but Lily was kind She gave it some carrots and the rabbit became her friend",
    "Tom was a brave boy He wanted to climb the big tree in his backyard He tried many times but he kept falling His father saw him and said You can do it Tom Tom tried one more time and he made it He was so happy",
    "There was a cat named Whiskers Whiskers liked to sleep on the warm sofa all day One morning Whiskers saw a bird outside the window The bird was singing a pretty song Whiskers watched the bird for a long time",
    "Emma and her mother went to the park Emma saw a big slide She wanted to go down the slide but she was scared Her mother said I will catch you Emma closed her eyes and went down the slide It was fun She did it again and again",
    "A little dog named Max ran after a ball in the garden The ball rolled under a bush Max pushed the bush with his nose The ball came out Max picked it up and ran back to his owner The owner said Good dog",
    "Ben wanted to build a sandcastle at the beach He had a bucket and a shovel He filled the bucket with sand and turned it over He made a tall tower Then he added a wall The sandcastle was beautiful",
    "Sara liked to draw pictures She drew a house with a red roof She drew a tree with green leaves She drew a sun with yellow rays Her mother looked at the picture and said This is wonderful Sara smiled",
    "The sun was shining The birds were singing It was a beautiful day Jack went outside to play He saw his friends at the park They played hide and seek Jack counted to ten and looked for his friends He found them all",
    "Anna had a new bicycle It was red and shiny She learned to ride it She fell a few times but she did not give up Soon she could ride without help She felt proud of herself",
    "A family of ducks lived near the pond The mother duck took her babies for a swim The baby ducks followed their mother in a line They learned to find food in the water They grew bigger every day",
    "Mike woke up early in the morning He looked out the window and saw snow Everything was white and quiet He put on his warm coat and boots He went outside to make a snowman He rolled three balls of snow and stacked them",
    "Lucy had a small garden behind her house She planted seeds in the soil Every day she watered them After a week little green plants started to grow Lucy was excited She watched them grow taller every day",
    "The old man lived in a small house by the lake Every morning he went fishing He sat quietly with his fishing rod He watched the water and the birds Sometimes he caught a fish Sometimes he just enjoyed the peace",
    "Two friends named Sam and Alex went on a hike They walked through the forest The trees were tall and green They saw a stream with clear water They drank the water and rested on a rock Then they continued their adventure",
    "A little bird fell from its nest It was too small to fly back up A girl named Mia found the bird She picked it up gently She put it back in the nest The mother bird sang a happy song to thank Mia",
    "Tim had a toy robot It could walk and talk Tim loved to play with his robot One day the robot stopped working Tim was sad His father helped him fix the robot They opened it and replaced the batteries The robot worked again",
    "The farmers worked in the fields all day They planted rice and vegetables The sun was hot but they kept working In the evening they went home tired but happy They ate dinner together and told stories about their day",
    "A little turtle lived in a pond He wanted to see the world outside the pond One day he climbed out and started walking He met a frog a rabbit and a deer They showed him many wonderful things But at night he missed his home",
    "Susan loved to read books She read stories about dragons and princesses and faraway lands Her favorite place was the library She could spend hours there reading one book after another Books took her to magical places",
    "The circus came to town There were clowns and acrobats and elephants Tom went to see the circus with his family The clowns made everyone laugh The acrobats did amazing tricks It was a night Tom would never forget",
    "A kind woman named Mrs Green lived next door to Peter She had a garden full of flowers Every morning she gave Peter fresh flowers for his mother Peter helped her water the plants They became good friends",
    "The forest animals had a meeting The squirrel said winter is coming we need to store food The bear said I will sleep through winter The birds said we will fly south They all prepared for winter in their own way",
    "Tom wanted to learn how to cook He asked his mother to teach him She showed him how to crack an egg and mix flour He made pancakes for breakfast They were not perfect but they tasted good Tom felt proud",
    "A little star in the sky felt lonely It looked down at the earth and saw children playing The star wished it could play too A wise moon said you shine bright for them at night That is your special gift The star smiled",
    "The baker woke up at four in the morning He mixed flour water and yeast He kneaded the dough and shaped it into loaves He put them in the oven The smell of fresh bread filled the street People came to buy his bread",
    "Kate found a map in her grandfathers attic The map showed a path to a hidden treasure She followed the map through the garden and behind the old tree She dug in the ground and found a box Inside were old coins and a letter",
    "The train station was busy People were going to different places Some were visiting family Some were going on holiday The train arrived with a loud whistle Everyone got on board The train traveled through mountains and fields",
    "A little seed lay under the ground It was dark and warm The seed waited and waited One day rain came and the seed drank the water The seed began to grow A small root went down and a small stem went up Soon a flower bloomed",
    "Mark and his sister built a fort in the living room They used blankets and chairs The fort had a roof and walls They brought pillows and books inside It was their secret hideout They played there all afternoon",
    "The river flowed through the village The children played near the river in summer They caught fish with nets They splashed in the water They built small boats from leaves and watched them float away The river was their playground",
]

# 额外训练文本（更长，用于丰富训练数据）
EXTRA_TRAINING_TEXT = """
The king and queen lived in a big castle The castle had tall towers and a large garden The princess loved to walk in the garden She picked flowers and talked to the birds One day a dragon came to the castle The people were scared but the princess was brave She talked to the dragon and found out it was lonely The dragon became their friend and protected the castle
The spaceship landed on a new planet The astronaut stepped out and looked around The sky was purple and the plants were blue He saw strange creatures with three eyes They were friendly and showed him their city He learned many new things and returned to earth with amazing stories
Tom and Jerry were best friends They did everything together They went to school together They played soccer together They even did homework together One day Jerry moved to a new city Tom was sad But they wrote letters to each other and stayed friends forever
"""

# BPE/TinyStories 词表（前几个 token 为特殊 token）
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3

# 从故事中自动构建词表
def build_vocab(stories, min_freq=1):
    """从故事列表构建词表"""
    from collections import Counter
    counter = Counter()
    for story in stories:
        counter.update(story.lower().split())
    # 过滤低频词
    vocab = {word for word, count in counter.items() if count >= min_freq}
    # 排序确保确定性
    sorted_vocab = sorted(vocab)
    word2idx = {
        "<pad>": PAD_TOKEN,
        "<bos>": BOS_TOKEN,
        "<eos>": EOS_TOKEN,
        "<unk>": UNK_TOKEN,
    }
    for i, word in enumerate(sorted_vocab):
        word2idx[word] = i + 4
    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx, idx2word


def encode(text, word2idx, max_len=64):
    """将文本转为 token ids"""
    words = text.lower().split()
    ids = [BOS_TOKEN]
    for w in words[:max_len - 2]:
        ids.append(word2idx.get(w, UNK_TOKEN))
    ids.append(EOS_TOKEN)
    # 填充
    if len(ids) < max_len:
        ids += [PAD_TOKEN] * (max_len - len(ids))
    return ids[:max_len]


def encode_prompt(text, word2idx, max_len=48):
    """将 prompt 编码为 token ids（不含 EOS/PAD）"""
    words = text.lower().split()
    ids = [word2idx.get(w, UNK_TOKEN) for w in words[:max_len - 1]]
    return torch.tensor([ids], dtype=torch.long)


def decode(ids, idx2word):
    """将 token ids 转回文本"""
    words = []
    for i in ids:
        if i == BOS_TOKEN:
            continue
        elif i == EOS_TOKEN:
            break
        elif i == PAD_TOKEN:
            continue
        words.append(idx2word.get(i, "<unk>"))
    return " ".join(words)


class TinyStoriesDataset(Dataset):
    """TinyStories 数据集"""
    def __init__(self, stories=None, word2idx=None, max_len=64):
        self.stories = stories or SAMPLE_STORIES
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        text = self.stories[idx]
        ids = encode(text, self.word2idx, self.max_len)
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


def create_dataloaders(stories=None, batch_size=4, max_len=64):
    """创建训练和验证 DataLoader"""
    if stories is None:
        stories = SAMPLE_STORIES + [s.strip() for s in EXTRA_TRAINING_TEXT.strip().split("\n") if s.strip()]
    word2idx, idx2word = build_vocab(stories)

    # 分割训练/验证
    split = int(len(stories) * 0.8)
    train_stories = stories[:split]
    val_stories = stories[split:]

    train_ds = TinyStoriesDataset(train_stories, word2idx, max_len)
    val_ds = TinyStoriesDataset(val_stories, word2idx, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader, word2idx, idx2word


def download_tinystories(target_path="tinystories.txt"):
    """尝试下载 TinyStories 数据集"""
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
    try:
        print(f"正在下载 TinyStories...")
        urllib.request.urlretrieve(url, target_path)
        print(f"下载完成: {target_path}")
        return target_path
    except Exception as e:
        print(f"下载失败（使用嵌入数据）: {e}")
        return None


def load_stories_from_file(path):
    """从文件加载故事列表"""
    import json
    stories = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    stories.append(data.get("text", ""))
                except json.JSONDecodeError:
                    stories.append(line)
    return stories if stories else SAMPLE_STORIES
