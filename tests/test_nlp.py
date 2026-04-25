import time
import sys
import os

# Ajustar o path para encontrar a pasta 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.search.nlp import TextProcessor 

def comparar_performance(processor, texto, idioma_codigo):
    """
    Testa as várias possibilidades de normalização para um determinado idioma.
    idioma_codigo: 'portuguese' ou 'english'
    """
    print(f"\n{'='*60}")
    print(f" 🌍 TESTANDO IDIOMA: {idioma_codigo.upper()}")
    print(f"{'='*60}")

    configuracoes = [
        {'name': 'Stemming', 'method': 'stem', 'sw': True},
        {'name': 'Lematização', 'method': 'lemma', 'sw': True},
        {'name': 'Apenas Tokenização (com SW)', 'method': None, 'sw': False},
    ]

    for config in configuracoes:
        print(f"\n▶ Executando: {config['name']}...")
        
        inicio = time.time()
        tokens = processor.process_text(
            texto, 
            language=idioma_codigo, 
            remove_stopwords=config['sw'], 
            normalization_method=config['method']
        )
        fim = time.time() - inicio
        
        vocab_unico = set(tokens)
        
        print(f"   ⏱  Tempo: {fim:.4f}s")
        print(f"   📦 Tokens gerados: {len(tokens)} (Únicos: {len(vocab_unico)})")
        print(f"   📝 Amostra: {tokens}...")

if __name__ == "__main__":
    # 1. Textos de Teste
    texto_pt = """
    Os computadores estão a aprender a processar as línguas humanas muito rapidamente. 
    Nós corremos vários testes todos os dias. As tecnologias de processamento avançaram imenso. 
    O investigador Silva encontrou várias soluções para os problemas técnicos.
    """

    texto_en = """
    Computers are learning to process human languages very quickly. 
    We ran several tests every day. Processing technologies have advanced immensely.
    Researcher Smith found several solutions for technical problems.
    """
    
    processador_unico = TextProcessor()
    print("✅ Modelos carregados com sucesso!\n")

    # Testar todas as possibilidades para Português
    comparar_performance(processador_unico, texto_pt, "portuguese")

    # Testar todas as possibilidades para Inglês
    comparar_performance(processador_unico, texto_en, "english")

    print(f"\n{'='*60}")
    print(" 🏁 TESTE DE TODAS AS POSSIBILIDADES CONCLUÍDO")
    print(f"{'='*60}")