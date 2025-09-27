from cs336_basics.pretokenization_example import find_chunk_boundaries_from_str
import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
import sys
import psutil
from collections import defaultdict
from collections.abc import Iterable
from typing import Iterator


def get_chunk_info(s: str, mem_ratio=0.05, min_chunk_mb=1, max_chunk_mb=100):
    """根据文本大小 & 可用内存，返回推荐块数和每块大小"""
    
    utf8_size = len(s.encode("utf-8"))  # 真实字节数
    available_mem = psutil.virtual_memory().available
    
    # 目标块大小 = 可用内存 * mem_ratio，限制在[min_chunk_mb, max_chunk_mb] MB
    target_chunk_size = min(
        max(int(available_mem * mem_ratio), min_chunk_mb * 1024 * 1024),
        max_chunk_mb * 1024 * 1024,
    )
    
    # 推荐块数（至少1块）
    num_chunks = max(1, (utf8_size + target_chunk_size - 1) // target_chunk_size)
    
    print(f"📊 文本大小: {utf8_size / 1024 / 1024:.2f} MB")
    print(f"💾 可用内存: {available_mem / 1024 / 1024:.2f} MB")
    print(f"🔧 推荐块大小: {target_chunk_size / 1024 / 1024:.2f} MB")
    print(f"📦 推荐拆分块数: {num_chunks}")
    
    return num_chunks, target_chunk_size
    
class tokenizer:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """读取 vocab.json 和 merges.txt，然后构造一个 BPETokenizer"""
        import json

        # 1. 读取 vocab
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab = json.load(f)
            vocab = {int(k): bytes(v) for k, v in vocab.items()}

        # 2. 读取 merges
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ")
                if len(parts) == 2:#parts是utf-8字符，解码成
                    merges.append((parts[0].decode("utf-8"), parts[1].decode("utf-8")))

        # 3. 构造 tokenizer 对象
        return cls(vocab, merges, special_tokens)



    def encode(self, text: str) -> list[int]:
        if text == "":
            return []
        num_chunks, _ = get_chunk_info(text)
        # 构建一个byte -> int 的反向字典 本来是int -> byte
        # 有些vocab
        reversed_vocab = {v:k for k,v in self.vocab.items()}
        # 更新next_idx
        # 默认special排到最后
        next_idx = 256
        
        # 构造merges_dict:将bytes映射为vocab中对应的int，只对merges中的规则进行映射，得到的便是编码，即(int,int)->int
        merges_dict = {}
        for (s1, s2) in self.merges:
            merges_dict[(reversed_vocab[s1], reversed_vocab[s2])] = next_idx
            next_idx += 1
            #只能给合并的部分的str->idx的映射，其他没有合并的只能先str-(utf-8)->byte-(vocab)->int

        # 分段处理
        char_boundaries = find_chunk_boundaries_from_str(text, num_chunks, self.special_tokens)
        chunks = [
            text[start:end] for start, end in zip(char_boundaries[:-1], char_boundaries[1:])
        ]

        output_list = []    #可以用numpy数组感觉

        for chunk in chunks:
            # 标准化换行
            chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")#长字符串
            
            if self.special_tokens: #有就要分割
                #pattern = "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
                pattern = "(" + "|".join(re.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True)) + ")"
                tokens_list = re.split(pattern, chunk) #不一定all_chunk中含义special_tokens
                tokens_list = [t for t in tokens_list if t != ""]#过滤空串
                print("tokesn_list:",tokens_list)
                for tokens in tokens_list:
                    #print("tokens:",tokens)
                    if tokens in self.special_tokens:
                        #print("ss:",tokens.encode("utf-8"))
                        token_idx_merged = reversed_vocab[tokens.encode("utf-8")]
                        output_list.append(token_idx_merged)
                    else:
                        for match in re.finditer(PAT, tokens):
                            token = match.group(0)
                            #print("处理：",[token])
                            ch_bytes = token.encode("utf-8")
                            token_idx = [reversed_vocab[bytes([b])] for b in ch_bytes]
                            #print("压缩编码为：",token_idx)
                            token_idx_merged = self._apply_bpe_merges(token_idx, merges_dict)
                            
                            #print("编码为：",token_idx_merged)
                            output_list.extend(token_idx_merged)
                        # 应用BPE合并
            else:
                for match in re.finditer(PAT, chunk):
                    # 如果没有特殊token，直接处理整个chunk
                    token = match.group(0)
                    ch_bytes=token.encode("utf-8")
                    token_idx = [reversed_vocab[bytes([b])] for b in ch_bytes]
                    token_idx_merged = self._apply_bpe_merges(token_idx, merges_dict)
                    output_list.extend(token_idx_merged)
            

            # 平展当前chunk的tokens并添加到输出
        
        return output_list
    def _apply_bpe_merges(self, token_idx: list[int], merges_dict: dict) -> list[int]:
        if len(token_idx) <= 1:
            return token_idx

        while True:
            # 1. 找出所有相邻 pair
            pairs = [(token_idx[i], token_idx[i+1]) for i in range(len(token_idx)-1)]

            # 2. 找出 merges_rank 里出现的 pair，并取 rank 最小的
            ranked_pairs = [(merges_dict[p], i, p) for i, p in enumerate(pairs) if p in merges_dict]
            if not ranked_pairs:
                break  # 没有可以合并的pair，结束

            _, idx, pair = min(ranked_pairs, key=lambda x: x[0])  # 取 rank 最小的 pair
            new_token_id = merges_dict[pair]

            # 3. 合并该 pair
            token_idx = token_idx[:idx] + [new_token_id] + token_idx[idx+2:]

        return token_idx



    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text_chunk in iterable:
            encoded_chunk = self.encode(text_chunk)
            for token_id in encoded_chunk:
                yield token_id


    def decode(self, ids: list[int]) -> str:
        #vocab:int->byde
        # decoded_pieces = []  # 用来收集解码后的每一段字符串
        
        # for token_id in ids:
        #     b = self.vocab.get(token_id)          # 1️⃣ 取出 bytes
        #     print(f"[DEBUG] token_id={token_id}, bytes={b}")

        #     try:
        #         s = b.decode("utf-8")             # 2️⃣ 解码成字符串
        #     except UnicodeDecodeError as e:
        #         print(f"[ERROR] 无法解码 token_id={token_id}, bytes={b}, 错误={e}")
        #         s = "�"  # 用替换符避免崩溃

        #     print(f"[DEBUG] 解码结果: {s!r}")
        #     decoded_pieces.append(s)

        # result = "".join(decoded_pieces)          # 3️⃣ 拼接
        # print(f"[DEBUG] 最终拼接结果: {result!r}")
        # return result
        if not ids:
            return ""
        # vocab
        byte_seq = b"".join(self.vocab[i] for i in ids)
        text = byte_seq.decode("utf-8")
        return text
        
if __name__ == "__main__":
    # 构造一个简易 vocab 和 merges 测试
    vocab = {i: bytes([i]) for i in range(256)}
    vocab.update({
        256: b"ll"      
    })
    vocab.update({
        257: "<|endoftext|>".encode("utf-8"),
        258: "<|endoftext|><|endoftext|>".encode("utf-8")
    })  

    # 假设把 ("l","l") 合并成新token 265-> bytes和 ->str
    merges = [(b"l", b"l")]

    tokenizer = tokenizer(vocab, merges, special_tokens=["<|endoftext|>","<|endoftext|><|endoftext|>"])

    # 测试文本
    #
    text = "Hello, how are you?🙃"
    text = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    text = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
    #text = "hello <|endoftext|> world!"
    print("!原始文本:", text)
    print("!原始文本逐字符的utf-8编码:",[ch.encode("utf-8") for ch in text])

    # 编码
    encoded = tokenizer.encode(text)
    print("!编码结果:", encoded)

    # 解码
    decoded = tokenizer.decode(encoded)
    print("!解码结果:", decoded)

    tokenized_string = [tokenizer.decode([x]) for x in encoded]
    print("!分步解码结果:", tokenized_string)
    # 流式输出
    #print("流式输出:", list(tokenizer.encode_iterable([text])))