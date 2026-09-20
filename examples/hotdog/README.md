# Not Hotdog

The SeeFood app from Silicon Valley, on jevify. One image, one question, one forward pass.

```bash
uv run --extra transformers python examples/hotdog/app.py --model kushalpatil/jevify-gemma4-e4b
open http://127.0.0.1:8000
```

Upload a photo, drop one in, click a sample, or start the camera and hit Live (a frame every
400 ms). You get the verdict, `P(hot dog)`, and what it actually is, from one call:

```python
jev.system_one({"image": img}, {
    "hotdog": Noul("Is this a hot dog?", {"true": "a sausage served in a sliced hot dog bun", "false": "anything else: a corn dog, a bare sausage, a sub, a burger, not food"}),
    "food":   Choice("What is shown in the image?", {f: None for f in FOODS}),
})
```

On the ten photos in `samples/` (jevify-gemma4-e4b, one B200, 130 to 450 ms per image):

| photo | verdict | P(hot dog) | actually |
|---|---|---|---|
| hotdog_mustard | HOTDOG | 0.998 | hot dog 0.99 |
| hotdog_chicago | HOTDOG | 0.995 | hot dog 0.95 |
| hotdog_stand | HOTDOG | 0.993 | hot dog 0.96 |
| corn_dog | NOT HOTDOG | 0.020 | corn dog 0.99 |
| sausage_no_bun | NOT HOTDOG | 0.023 | sausage without bun 0.97 |
| burger | NOT HOTDOG | 0.002 | hamburger 1.00 |
| pizza | NOT HOTDOG | 0.000 | pizza 1.00 |
| banana | NOT HOTDOG | 0.000 | banana 1.00 |
| dachshund | NOT HOTDOG | 0.001 | an animal 0.95 |
| sub_sandwich | HOTDOG | 0.706 | sub sandwich 0.59, hot dog 0.31 |

The sub is the miss, and the number says so: 0.71, not 0.99. A raw Gemma 4 gives 0.99 on
everything it calls a hot dog.

Runs on a Mac with MPS too, at a few seconds per image. Sample photos are from Wikimedia Commons.
