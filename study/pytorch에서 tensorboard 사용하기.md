# Pytorch에서 Tensorboard 사용하기



<br/>

### 주의!

```
from torch.utils.tensorboard import SummaryWriter
```

를 했을때 아래와 같은 error가 나올 경우

```
no module named tensorboard
```

<br/>

````
pip install tensorboard
````

를 사용해서 tensorboard를 설치해줘야 한다.





<br/>

## Tensorboard 설정

**torch.utils**의 **tensorboard**를 불러오기

Tensorboard에 정보를 제공(write)하는 **SummaryWriter** 정의

```python
from torch.utils.tensorboard import SummaryWriter

# 기본 log_dir은 runs 폴더이다.
writer = SummaryWriter()
```



<br/>

`writer.add_image()`, `writer.add_scalars()` 등이 있다.





<br/>

- 예시 코드

```python
from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter()

for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
```



<br/>

- run

```
tensorboard --logdir runs
```

이때, `runs`는 자동으로 생성된 폴더 이름이다.

writer를 선언할때 변경해줄 수 있다.

기본 포트: 6006

<br/>

```
localhost:6006
```





<br/>

<br/><br/>

-------------

### Reference

- https://tutorials.pytorch.kr/intermediate/tensorboard_tutorial.html

- https://keep-steady.tistory.com/14
- https://gaussian37.github.io/dl-pytorch-observe/