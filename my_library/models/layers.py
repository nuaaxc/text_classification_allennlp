from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding, PretrainedBertEmbedder


def bert_embeddings(pretrained_model: str,
                    training: bool = False,
                    top_layer_only: bool = True
                    ) -> BasicTextFieldEmbedder:
    """Pre-trained embeddings using BERT"""
    bert = PretrainedBertEmbedder(
        requires_grad=training,
        pretrained_model=pretrained_model,
        top_layer_only=top_layer_only
    )
    word_embeddings = BasicTextFieldEmbedder(
        token_embedders={'tokens': bert},
        embedder_to_indexer_map={'tokens': ['tokens', 'tokens-offsets']},
        allow_unmatched_keys=True)
    return word_embeddings
