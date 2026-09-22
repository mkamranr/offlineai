# simple-python

The smallest useful package: local files, carried across an air gap and
verified on arrival. No container, no GPU, no network at any point after the
build.

```bash
offlineai build examples/simple-python
offlineai verify simple-python-1.0.0.offlineai
offlineai import simple-python-1.0.0.offlineai
offlineai info simple-python
```

Use this shape when you have weights produced in-house that never lived in a
public hub, and you need them moved with checksums and a manifest rather than
by hand.
