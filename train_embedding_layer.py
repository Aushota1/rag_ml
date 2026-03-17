#!/usr/bin/env python3
"""
Обучение Embedding Layer для токенизатора
Создает embedding_model.pth для использования в классификаторах
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from parser import DocumentParser
from tokenier_integration.bpe_tokenizer import BPETokenizer
from tokenier_integration.embedding_layer import EmbeddingLayer


def load_tokenizer(tokenizer_path: str):
    """Загрузка обученного токенизатора"""
    print(f"Loading tokenizer from: {tokenizer_path}")
    
    if not Path(tokenizer_path).exists():
        print(f"Error: Tokenizer not found: {tokenizer_path}")
        print("Please train tokenizer first: python train_bpe_tokenizer.py")
        sys.exit(1)
    
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    print(f"✓ Tokenizer loaded (vocab size: {tokenizer.get_vocab_size()})")
    return tokenizer


def load_documents_from_pdfs(documents_path: str):
    """Загрузка текстов из PDF документов"""
    print(f"\nLoading documents from: {documents_path}")
    docs_path = Path(documents_path)
    
    if not docs_path.exists():
        print(f"Error: Path not found: {docs_path}")
        sys.exit(1)
    
    pdf_files = list(docs_path.glob("*.pdf"))
    if not pdf_files:
        print(f"Error: No PDF files found")
        sys.exit(1)
    
    print(f"Found {len(pdf_files)} PDF files")
    parser = DocumentParser()
    texts = []
    
    for pdf_file in tqdm(pdf_files, desc="Parsing PDFs"):
        try:
            doc = parser.parse_pdf(pdf_file)
            if doc and doc.get('text'):
                text = doc['text'].strip()
                if len(text) > 100:
                    texts.append(text)
        except Exception as e:
            continue
    
    print(f"Successfully loaded {len(texts)} documents")
    return texts


def create_training_data(texts, tokenizer, max_seq_len=512):
    """Создание обучающих данных"""
    print("\nCreating training data...")
    sequences = []
    pad_id = tokenizer.special_tokens.get('<PAD>', 0)
    
    for text in tqdm(texts, desc="Tokenizing"):
        token_ids = tokenizer.encode(text)
        for i in range(0, len(token_ids), max_seq_len):
            seq = token_ids[i:i + max_seq_len]
            if len(seq) < max_seq_len:
                seq = seq + [pad_id] * (max_seq_len - len(seq))
            sequences.append(seq)
    
    print(f"Created {len(sequences)} sequences")
    return sequences


class NextTokenPredictionModel(nn.Module):
    """Модель для обучения эмбеддингов"""
    def __init__(self, embedding_layer, hidden_dim=512):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.lstm = nn.LSTM(
            embedding_layer.embedding_dim, hidden_dim, 
            num_layers=2, batch_first=True, dropout=0.1
        )
        self.output = nn.Linear(hidden_dim, embedding_layer.vocab_size)
    
    def forward(self, token_ids):
        embeddings = self.embedding_layer(token_ids)
        lstm_out, _ = self.lstm(embeddings)
        return self.output(lstm_out)



def train_embedding_layer(tokenizer, texts, embedding_dim=256, max_seq_len=512,
                          hidden_dim=512, batch_size=16, num_epochs=3, 
                          learning_rate=0.001, output_path="models/tokenier/embedding_model.pth",
                          checkpoint_path="models/tokenier/checkpoint.pkl"):
    """Обучение embedding layer"""
    print("\n" + "=" * 70)
    print("Training Embedding Layer")
    print("=" * 70)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sequences = create_training_data(texts, tokenizer, max_seq_len)
    
    vocab_size = tokenizer.get_vocab_size()
    padding_idx = tokenizer.special_tokens.get('<PAD>', 0)
    
    embedding_layer = EmbeddingLayer(
        vocab_size=vocab_size, embedding_dim=embedding_dim,
        max_seq_len=max_seq_len, dropout=0.1, padding_idx=padding_idx,
        learnable_pos=False, layer_norm=True
    )
    
    model = NextTokenPredictionModel(embedding_layer, hidden_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)
    
    print("\nStarting training (Press Ctrl+C to stop and save)")
    
    try:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            num_batches = 0
            
            pbar = tqdm(range(0, len(sequences), batch_size), 
                       desc=f"Epoch {epoch + 1}/{num_epochs}")
            
            for i in pbar:
                batch = sequences[i:i + batch_size]
                input_ids = torch.tensor(batch, dtype=torch.long).to(device)
                target_ids = torch.roll(input_ids, -1, dims=1)
                target_ids[:, -1] = padding_idx
                
                optimizer.zero_grad()
                logits = model(input_ids)
                loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}")
            
            checkpoint = {
                'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_loss,
                'embedding_dim': embedding_dim, 'max_seq_len': max_seq_len,
                'hidden_dim': hidden_dim, 'vocab_size': vocab_size
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Checkpoint saved")
    
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted - saving progress...")
        checkpoint = {
            'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / max(num_batches, 1),
            'embedding_dim': embedding_dim, 'max_seq_len': max_seq_len,
            'hidden_dim': hidden_dim, 'vocab_size': vocab_size
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"✅ Saved to: {checkpoint_path}")
    
    print(f"\nSaving embedding layer to: {output_path}")
    embedding_state = {
        'embedding_dim': embedding_dim, 'vocab_size': vocab_size,
        'max_seq_len': max_seq_len, 'padding_idx': padding_idx,
        'state_dict': embedding_layer.state_dict()
    }
    torch.save(embedding_state, output_path)
    print("✅ Training completed!")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train embedding layer")
    parser.add_argument('--tokenizer-path', default='models/tokenier/tokenizer.pkl')
    parser.add_argument('--documents-path', default='C:/Users/Aushota/Downloads/dataset_documents')
    parser.add_argument('--embedding-dim', type=int, default=256)
    parser.add_argument('--max-seq-len', type=int, default=512)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--output', default='models/tokenier/embedding_model.pth')
    parser.add_argument('--checkpoint', default='models/tokenier/checkpoint.pkl')
    args = parser.parse_args()
    
    print("=" * 70)
    print("EMBEDDING LAYER TRAINING")
    print("=" * 70)
    
    tokenizer = load_tokenizer(args.tokenizer_path)
    texts = load_documents_from_pdfs(args.documents_path)
    
    if not texts:
        print("Error: No texts loaded")
        sys.exit(1)
    
    train_embedding_layer(
        tokenizer=tokenizer, texts=texts, embedding_dim=args.embedding_dim,
        max_seq_len=args.max_seq_len, hidden_dim=args.hidden_dim,
        batch_size=args.batch_size, num_epochs=args.num_epochs,
        learning_rate=args.learning_rate, output_path=args.output,
        checkpoint_path=args.checkpoint
    )
    
    print("\n✅ DONE!")
    print(f"Next: python train_tokenier_models.py")


if __name__ == "__main__":
    main()
