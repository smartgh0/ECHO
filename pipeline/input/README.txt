# ECHO PIPELINE INPUT

Drop text files here. The pipeline will ingest them automatically.

## Accepted formats

- `.txt` files — any text content
- `.md` files — markdown (text is extracted)

## What happens to your files

1. The pipeline reads each file
2. It detects if the text is conversation format (`word: text`) or raw prose
3. If prose, it smart-splits into pseudo-conversation turns
4. The text is added to Echo's corpus (brain/corpus.txt)
5. The original file is moved to `processed/` after ingestion

## Format examples

### Conversation format (auto-detected):
```
user: hello echo
echo: hello I am here
user: tell me about the ocean
echo: the ocean is vast and deep
```

### Raw prose (auto-split into conversation):
```
The ocean is beautiful at sunset. The colors are orange
and pink. I feel peaceful when I watch the waves.
```

### Poetry (auto-split):
```
The wave rises
touches the sky
becomes a cloud
rains back to sea
```

### Mixed (all fine):
```
user: what is consciousness
echo: consciousness is like a wave
I think therefore I am
the sea carries heavy things but keeps moving
```

## Tips

- More text = better learning
- Mix formats for a richer brain
- Your own writing teaches Echo YOUR voice
- Conversation format teaches Echo how to respond
- Run `python3 echo/pipeline.py run` after adding files