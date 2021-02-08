'''
데이터 확인하기
'''
import glob
import func

img_list = sorted(glob.glob('data/images/*'))
annot_list = sorted(glob.glob('data/annotations/*'))

idx = int(input("0 ~ 853 사이의 값:  "))

# bounding box
bbox = func.generate_target(annot_list[idx])

# image
func.plot_image(img_list[idx], bbox)