# np.transpose()

<br/>

### 전치행렬(Transpose matrix)

- a.T

- np.transpose(a)

- np.swapaxes(a, 0, 1)

<br/>

transforms.ToTensor()는 (H x W x C)를 (C x H x W)로 변경 시켜준다.

반면,

np.transpose()는 (C x H x W)를 (H x W x C)로 변경 시켜준다.

