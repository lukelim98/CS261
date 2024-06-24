import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from gtgpt.model import DummyMultiHeadedSelfAttention, DummyBlock, DummyTransformer, DummyEmbedding
from gtgpt.utils import set_seed
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)

class Embedding(DummyEmbedding):
    def forward(self, idx):
        """
        :param idx: intTensor of shape (B,T)
        :returns embeddings: floatTensor of shape (B,T,n_embd)
        """
        B, T = idx.size()
        embeddings = None
        #############################################################################
        # TODO:
        # Implement the embedding lookup.                                           #
        #                                                                           #
        # This will take a few lines.                                               #
        #############################################################################
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        # Step 2: Retrieve token embeddings
        token_embeddings = self.vocab_embeddings(idx)  # (B, T, n_embd)

        # Step 3: Generate position indices
        position_indices = torch.arange(T, device=idx.device).expand(B, T)  # (B, T)

        # Step 4: Retrieve positional embeddings
        position_embeddings = self.position_embeddings(position_indices)  # (B, T, n_embd)

        # Step 5: Combine token and positional embeddings
        embeddings = token_embeddings + position_embeddings  # (B, T, n_embd)
        return embeddings

class GenericSelfAttention(DummyMultiHeadedSelfAttention):
    def forward(self, x, attention_mask):
        """
        :param x: float Tensor of shape (batch size, sequence length, embedding dimensionality)
        :param attention_mask: int Tensor of shape (batch size, 1, sequence length, sequence_length)
        :returns y: float Tensor of shape (batch size, sequence length, embedding dimensionality)
        """
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        y = None

        #############################################################################
        # TODO:                                                                     #
        # Implement multi-headed self-attention in GPT-2 Style                      #
        # Use the provided layers initialized in the DummySelfAttention constructor #
        # Apply dropout to the attention values after softmax and the final output  #
        #                                                                           #
        # Reference:                                                                #
        # https://jalammar.github.io/illustrated-gpt2/#part-2-illustrated-self-attention
        #                                                                           #
        # Note: All heads should be computed in parallel using the q,k,v layers     #
        #                                                                           #
        # For each item in the batch, if attention_mask[b, i, j] = 0,               #
        # then you should manually set the attention from token i to j to be -inf   #
        # Hint: See torch.masked_fill                                               #
        #############################################################################
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        d_k = C // self.n_head

        # Prepare Q, K, V
        Q = self.q(x).view(B, T, self.n_head, d_k).transpose(1, 2) # (B, n_head, T, d_k)
        K = self.k(x).view(B, T, self.n_head, d_k).transpose(1, 2)
        V = self.v(x).view(B, T, self.n_head, d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5) # (B, n_head, T, T)
        scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to V
        y = torch.matmul(attn, V) # (B, n_head, T, d_k)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)

        # Project back to the embedding dimension
        y = self.c_proj(y)
        y = self.hidden_dropout(y)
        # print("Generic Self Attention shape: ", y.shape)
        return y

#@title Now, we can very simply create a single layer transformer block!
class TransformerBlock(DummyBlock):
    def __init__(self, config):
        super().__init__(config, GenericSelfAttention)

    # A Basic Transformer Block with Attention followed by an MLP
    # note the layer norms and residual information preserved at each step.
    def forward(self, x, attention_mask):
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlpf(self.ln_2(x))
        return x

class GenericTransformer(DummyTransformer):
    def __init__(self, config):
        super().__init__(config, TransformerBlock, Embedding)
        self.block_size = config.block_size # Maximum Number of Tokens which can be encoded at once
        self.vocab_size = config.vocab_size

    def get_attention_mask(self, num_tokens):
        """
        Dummy For now, we will see how we use this later!
        """
        B = num_tokens.shape[0]
        return torch.ones((B, self.block_size, self.block_size))[:, :num_tokens.max().item(), :num_tokens.max().item()]

    def forward(self, idx, targets=None, hidden_cache=None, return_hidden=False):
        """
        :param idx: int Tensor of shape (B,T)
        :param hidden_cache: float Tensor of shape (B,P_T,n_embd)
        :param targets: int Tensor of shape (B,T_T)
        :param return_hidden: bool
        (if return_hidden = None)
        :returns x: float Tensor of shape (B,T,n_embd)
        (else)
        :returns logits: float Tensor of shape (B, T, vocab_size)
        :returns loss: float Tensor of shape (B) or None
        """
        num_tokens = (idx != -1).type(torch.int).sum(dim=1)
        if hidden_cache is not None:
          num_tokens = num_tokens + hidden_cache.shape[1]
        idx = idx.masked_fill(idx == -1, int(0)).type(torch.int)[:, :num_tokens.max().item()]
        if targets is not None:
          targets = targets[:, :num_tokens.max().item()]
        attention_mask = self.get_attention_mask(num_tokens)
        #############################################################################
        # TODO:                                                                     #
        # Put all the modules of a Transformer together for inference               #
        #                                                                           #
        # If hidden_cache exists,                                                   #
        # then the Transformer inputs should be concatenated in the token dimension #
        # First) All Embeddings from Hidden Cache                                   #
        # Next)  All Embeddings of tokens from idx.                                 #
        #                                                                           #
        # All the modules you'll need are listed above!                              #
        #                                                                           #
        # Note: You can iterate through a nn.ModuleList using a standard for loop.  #
        #                                                                           #
        # This will take a few lines!                                               #
        ##############################################################################
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        
        # Embed tokens
        x = self.transformer['embedding'](idx)
        if hidden_cache is not None:
            x = torch.cat([hidden_cache, x], dim=1)
        # Apply transformer blocks
        for block in self.transformer['h']:
            x = block(x, attention_mask)
            # Apply final layer normalization
            x = self.transformer['ln_f'](x)

        # Get logits
        logits = self.lm_head(x)
        
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            s_logits = logits
            if hidden_cache is not None:
              s_logits = logits[:, hidden_cache.shape[1]-1:-1].contiguous()
            loss = F.cross_entropy(
                s_logits.reshape(-1, self.vocab_size), targets.reshape(-1), ignore_index=-1
            )
        if return_hidden:
            return x

        return logits, loss

class Encoder(GenericTransformer):
    """Encoder Style Transformer with Bidirectional Attention"""
    def get_attention_mask(self, num_tokens):
        """
        :param num_tokens: int Tensor of shape (batch size)
        :returns attention_mask: int tensor of shape (batch_size, 1, max_tokens, max_tokens)
        """
        B = num_tokens.shape[0]
        max_tokens = min(self.block_size, num_tokens.max().item())
        ##############################################################################
        # TODO:                                                                      #
        # Implement a padding mask function.                                         #
        # This allows batching sequences of different lengths.                       #
        #                                                                            #
        # For example, for any row attention_mask[b, i] the following should be true:#
        #               For j < num_tokens[b], attention_mask[b, i, j] = 1          #
        #               For j >= num_tokens[b],  attention_mask[b, i, j] = 0         #
        #                                                                            #
        # Reference:https://huggingface.co/docs/transformers/glossary#attention-mask #                                                                #
        #                                                                            #
        # This should be a 1-3 line function.                                        #
        ##############################################################################
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        # Initialize the mask with zeros
        attention_mask = torch.zeros(B, 1, max_tokens, max_tokens, dtype=torch.int)

        # Fill the mask
        for b in range(B):
            valid_tokens = num_tokens[b].item()
            attention_mask[b, :, :valid_tokens, :valid_tokens] = 1
        # print('Encoder output shape: ', attention_mask.reshape(B, 1, max_tokens, max_tokens).shape)
        return attention_mask

class Decoder(Encoder):
    """Decoder Style model with a Causal Attention Mask"""

    def get_attention_mask(self, num_tokens):
        """
        :param num_tokens: int Tensor of shape (batch size)
        :returns attention_mask: int tensor of shape (batch_size, 1, block_size, block_size)
        """
        full_attention_mask = super().get_attention_mask(num_tokens)
        ##############################################################################
        # TODO:                                                                      #
        # Modify the output of the full encoder mask to create a "causal" mask       #
        # such that tokens only attend to tokens which occured earlier in the input. #
        #                                                                            #
        # For example, for any row attention_mask[b, i} the following should be true:#
        #               For j <= i, attention_mask[b, i, j] = 1                      #
        #               For j > i,  attention_mask[b, i, j] = 0                      #
        #                                                                            #
        # This should be a one line function which modifies the full attention_mask  #
        ##############################################################################
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        B, _, max_tokens, _ = full_attention_mask.shape
        full_attention_mask = torch.zeros(B, max_tokens, dtype=torch.int, device=num_tokens.device)
        for b in range(B):
            full_attention_mask[b, :num_tokens[b]] = 1

        # Generate a causal mask
        causal_mask = torch.triu(torch.ones((max_tokens, max_tokens), device=num_tokens.device), diagonal=1).bool()

        # Apply the causal mask on the full attention mask
        attention_mask = full_attention_mask.unsqueeze(1).unsqueeze(2).repeat(1, 1, max_tokens, 1)
        attention_mask = attention_mask.masked_fill(causal_mask, 0)
        # print("Decoder output shape: ", attention_mask.shape)
        return attention_mask

def generate(model, idx, max_new_tokens, temperature=1.0):
    """
    :param idx: int Tensor of shape (B, T)
    :param max_new_tokens: int
    :param temperature: Float
    :returns idx: int Tensor of shape (B, T+max_new_tokens)
    """
    ##############################################################################
    # TODO:                                                                      #
    # Sample from your model max_new_tokens times                                #
    # You should feed the predictions back into the model each time              #
    #                                                                            #
    # Adjust the probability distribution to be more or less greedy using        #
    # the temperature parameter                                                  #
    #                                                                            #
    # Reference: https://huggingface.co/blog/how-to-generate#sampling            #
    # Temperature Reference:                                                     #
    # https://web.stanford.edu/class/cs224n/slides/cs224n-2023-lecture10-nlg.pdf#page=34 #
    ##############################################################################
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    with torch.no_grad():  # No need to track gradients
        for _ in range(max_new_tokens):
            logits, _ = model(idx)  # Get the logits from the model
            # Take the logits of the last predicted token
            logits = logits[:, -1, :] / temperature
            # Apply softmax to get probabilities
            probabilities = torch.softmax(logits, dim=-1)
            # Sample a token from the probability distribution
            next_token = torch.multinomial(probabilities, num_samples=1)
            # Append the sampled token to the input
            idx = torch.cat((idx, next_token), dim=1)
    return idx

class EncoderDecoder(nn.Module):
    """Encoder-Decoder Model which combines the two architectures"""
    def __init__(self, encoder_config, decoder_config):
        super().__init__()
        # Add end of sequence token.
        decoder_config.vocab_size += 1
        self.vocab_size = decoder_config.vocab_size
        self.encoder = Encoder(encoder_config)
        self.decoder = Decoder(decoder_config)

    def configure_optimizers(self, train_config):
        enc_groups = self.encoder.configure_optimizers(train_config)
        dec_groups = self.decoder.configure_optimizers(train_config)
        return enc_groups + dec_groups

    def forward(self, prefix, targets=None):
        """
        :param prefix: int Tensor of shape (B,P_T)
        :param idx: float Tensor of shape (B,P_T,n_embd)
        :returns logits: float Tensor of shape (B, vocab_size)
        :returns loss: float Tensor of shape (B) or None
        """
        B = prefix.shape[0]
        idx = torch.tensor([[]]).repeat(B, 1)
        if targets is not None:
          idx = torch.cat((idx, targets), dim=1)

        ##############################################################################
        # TODO:                                                                      #
        # Create an Encoder Decoder Model by combining your previous transformers    #
        # The Encoder should encode the tokens from prefix into an embeddings        #
        # Use these in the hidden_cache to condition decoder generation              #
        #                                                                            #
        # This should be a 1-2 lines.                                                #
        ##############################################################################
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        encoded_prefix = self.encoder(prefix, return_hidden=True)
        logits, loss = self.decoder(idx, hidden_cache=encoded_prefix, targets=targets)
        return logits, loss

def prefix_generate(model, prefix, max_new_tokens, temperature=1.0):
    # Ensure the model is in evaluation mode
    model.eval()
    
    
    idx = torch.tensor([[]], dtype=torch.long)
    
    # Perform generation iteratively
    with torch.no_grad():
        # Encode the prefix using the encoder
        hidden_cache = model.encoder(prefix, return_hidden=True)
        for _ in range(max_new_tokens):
            # Decode the current sequence to generate the next token
            logits, _ = model.decoder(idx, hidden_cache=hidden_cache)
            # Process logits of the last generated token
            logits = logits[:, -1, :] / temperature
            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            # Append the next token to the current sequence
            idx = torch.cat([idx, next_token], dim=1)
    # Return only the newly generated part of the sequence
    return idx
