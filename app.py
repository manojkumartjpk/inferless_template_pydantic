import json
import os
import numpy as np
import torch
from transformers import pipeline
from datetime import datetime

import inferless
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_NAME = "EleutherAI/gpt-neo-125m"

@inferless.request
class RequestObjects(BaseModel):
    TEXT: str = Field(default="a horse near a beach")


@inferless.response
class ResponseObjects(BaseModel):
    OUT: str = Field(default='Test output')

@inferless.config()
class Config(BaseModel):
    is_streaming_output: bool = True

class InferlessPythonModel:

    def initialize(self):
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

    async def infer(self, inputs, stream_output_handler):
        prompt = inputs.TEXT
        
        # GPT-Neo doesn't use chat format; it's just a plain prompt
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        
        generation_kwargs = dict(
            inputs=input_ids,
            streamer=self.streamer,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            repetition_penalty=1.2,
            max_new_tokens=256,
        )
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in self.streamer:
            output_dict = {"OUT": new_text}
            await stream_output_handler.send_streamed_output(output_dict)
        thread.join()

        await stream_output_handler.finalise_streamed_output()

    def finalize(self):
        self.model = None
        self.tokenizer = None
