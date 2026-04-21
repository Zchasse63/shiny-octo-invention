# smoke_test_v3_anthropic

**When:** 2026-04-21T20:06:22+00:00 UTC
**Provider:** anthropic / `anthropic/default`
**Tokens:** input=23 output=108
**Estimated cost:** $0.0017

## Question

In 2 sentences: what is vbt.cv_split and how does it differ from using vbt.Splitter directly?

## Answer

`vbt.cv_split` is a decorator/helper that runs a function or pipeline through cross-validation using a splitter, and it automatically selects the best parameter set on the training data before applying it to the test data. In contrast, `vbt.Splitter` is just the splitter object itself—you create and pass it around directly, and if you use it without `cv_split`, you handle the split application and any parameter selection manually.
