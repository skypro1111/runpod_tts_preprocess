import base64
import os
import torch
import torchaudio
import numpy as np
import onnxruntime
import re
import random
import math
import pickle
import hashlib
from typing import Dict, Union, List, Tuple
from pypinyin import lazy_pinyin, Style
from dotenv import load_dotenv, find_dotenv
import runpod

# Завантаження змінних середовища
load_dotenv(find_dotenv('.env_prod'))

# Константи
HOP_LENGTH = 256
SAMPLE_RATE = 24000
RANDOM_SEED = random.randint(0, 1000000)

# Шляхи до моделей та файлів
ONNX_MODEL_PATH = os.getenv("F5_Preprocess")
VOCAB_FILE = os.getenv("vocab_file")

if not ONNX_MODEL_PATH or not VOCAB_FILE:
    raise ValueError("F5_Preprocess or vocab_file not found in environment variables")

# Завантаження словника
with open(VOCAB_FILE, "r", encoding="utf-8") as f:
    VOCAB_CHAR_MAP = {char[:-1]: i for i, char in enumerate(f)}

# Налаштування ONNX сесії
SESSION_OPTS = onnxruntime.SessionOptions()
SESSION_OPTS.log_severity_level = 3  # error level
SESSION_OPTS.inter_op_num_threads = 0  # Run different nodes with num_threads. Set 0 for auto.
SESSION_OPTS.intra_op_num_threads = 0  # Under the node, execute the operators with num_threads. Set 0 for auto.
SESSION_OPTS.enable_cpu_mem_arena = True  # True for execute speed
SESSION_OPTS.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
SESSION_OPTS.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
SESSION_OPTS.enable_mem_pattern = True
SESSION_OPTS.enable_mem_reuse = True
SESSION_OPTS.add_session_config_entry("session.intra_op.allow_spinning", "1")
SESSION_OPTS.add_session_config_entry("session.inter_op.allow_spinning", "1")
SESSION_OPTS.add_session_config_entry("arena_extend_strategy", "kSameAsRequested")
onnxruntime.set_seed(RANDOM_SEED)

def convert_char_to_pinyin(text_list: Union[List[str], List[List[str]]], polyphone: bool = True) -> List[List[str]]:
    """Конвертує текст в піньїнь"""
    final_text_list = []

    def replace_quotes(text: str) -> str:
        return text.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'").replace(';', ',')

    def is_japanese(c: str) -> bool:
        return (
                '\u3040' <= c <= '\u309F' or  # Hiragana
                '\u30A0' <= c <= '\u30FF' or  # Katakana
                '\uFF66' <= c <= '\uFF9F'  # Half-width Katakana
        )

    for text in text_list:
        char_list = []
        text = replace_quotes(text)
        
        for seg in text:
            seg_byte_len = len(seg.encode('utf-8'))
            if seg_byte_len == len(seg):  # ASCII text
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # Pure Chinese text
                seg_pinyin = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for p in seg_pinyin:
                    if p not in "。，、；：？！《》【】—…":
                        if not char_list or not is_japanese(char_list[-1]):
                            char_list.append(" ")
                    char_list.append(p)
            else:  # Mixed text or other languages
                for c in seg:
                    if ord(c) < 256:  # ASCII character
                        char_list.append(c)
                    elif is_japanese(c):  # Japanese character
                        char_list.append(c)
                    else:
                        if c not in "。，、；：？！《》【】—…":
                            pinyin = lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True)
                            char_list.extend(pinyin)
                        else:
                            char_list.append(c)
        final_text_list.append(char_list)
    return final_text_list

def list_str_to_idx(
    text: Union[List[str], List[List[str]]],
    vocab_char_map: Dict[str, int],
    padding_value: int = -1
) -> torch.Tensor:
    """Конвертує список строк в тензор індексів"""
    get_idx = vocab_char_map.get
    list_idx_tensors = [torch.tensor([get_idx(c, 0) for c in t], dtype=torch.int32) for t in text]
    text = torch.nn.utils.rnn.pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text

def generate_cache_key(audio_data: bytes, ref_text: str) -> str:
    """Generate a unique cache key based on reference audio and text"""
    audio_hash = hashlib.md5(audio_data).hexdigest()
    ref_text_hash = hashlib.md5(ref_text.encode('utf-8')).hexdigest()
    return f"{audio_hash}_{ref_text_hash}"

def process_reference_input(
    reference_audio: bytes,
    ref_text: str
) -> Tuple:
    """
    Process reference audio and text only, returning the preprocessed tensors
    
    Args:
        reference_audio: Binary audio data
        ref_text: Reference text
        
    Returns:
        Tuple: Reference audio tensors and text IDs
    """
    # Write audio data to a temporary file for torchaudio to process
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_file.write(reference_audio)
        temp_file_path = temp_file.name
    
    try:
        # Load and process audio
        audio, sr = torchaudio.load(temp_file_path)
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            audio = resampler(audio)
        audio = audio.unsqueeze(0).numpy()
        
        # Get model type
        ort_session = onnxruntime.InferenceSession(
            ONNX_MODEL_PATH,
            sess_options=SESSION_OPTS,
            providers=['CUDAExecutionProvider']
        )
        model_type = ort_session._inputs_meta[0].type
        
        # Convert audio to float16 if needed
        if "float16" in model_type:
            audio = audio.astype(np.float16)
            
        # Process reference text
        ref_text_processed = convert_char_to_pinyin([ref_text])
        ref_text_ids = list_str_to_idx(ref_text_processed, VOCAB_CHAR_MAP).numpy()
        
        # Calculate reference audio length
        ref_audio_len = audio.shape[-1] // HOP_LENGTH + 1
        
        # Package results
        result = (audio, ref_text_ids, ref_audio_len)
        return result
    
    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)

def preprocess_reference(
    reference_audio: bytes,
    ref_text: str
) -> bytes:
    """
    Preprocesses reference audio and text, optimized for later combination with generation text
    
    Args:
        reference_audio: Binary audio data
        ref_text: Reference text
        
    Returns:
        bytes: Serialized preprocessed reference data
    """
    # Process reference input
    audio, ref_text_ids, ref_audio_len = process_reference_input(reference_audio, ref_text)
    
    # Create ONNX session
    ort_session = onnxruntime.InferenceSession(
        ONNX_MODEL_PATH,
        sess_options=SESSION_OPTS,
        providers=['CUDAExecutionProvider']
    )
    
    # Get input/output names
    in_names = [input.name for input in ort_session.get_inputs()]
    out_names = [output.name for output in ort_session.get_outputs()]
    
    # Calculate maximum duration (just reference length for preprocessing)
    max_duration = np.array(ref_audio_len, dtype=np.int64)
    
    # Run ONNX inference
    outputs = ort_session.run(
        out_names,
        {
            in_names[0]: audio,
            in_names[1]: ref_text_ids,
            in_names[2]: max_duration
        }
    )
    
    # Create time_expand tensor
    t = torch.linspace(0, 1, 32 + 1, dtype=torch.float32)
    time_step = t + (-1.0) * (torch.cos(torch.pi * 0.5 * t) - 1 + t)
    delta_t = torch.diff(time_step)
    
    time_expand = torch.zeros((1, 32, 256), dtype=torch.float32)
    half_dim = 256 // 2
    emb_factor = math.log(10000) / (half_dim - 1)
    emb_factor = 1000.0 * torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb_factor)
    
    for i in range(32):
        emb = time_step[i] * emb_factor
        time_expand[:, i, :] = torch.cat((emb.sin(), emb.cos()), dim=-1)
    
    # Package result
    result = {
        "preprocessed": (
            outputs[0],  # noise
            outputs[3],  # cat_mel_text (cond)
            outputs[4],  # cat_mel_text_drop (cond_drop)
            time_expand.numpy(),  # time_expand
            outputs[1],  # rope_cos
            outputs[2],  # rope_sin
            delta_t.numpy(),  # delta_t
            outputs[6]   # ref_signal_len
        ),
        "ref_data": {
            "audio": audio,
            "text_ids": ref_text_ids,
            "audio_len": ref_audio_len
        }
    }
    
    # Serialize result
    return pickle.dumps(result)

def handler(event):
    print(event)  
    input = event["input"]
    audio = base64.b64decode(input["reference_audio"])
    return preprocess_reference(audio, input["ref_text"])


runpod.serverless.start({"handler": handler})