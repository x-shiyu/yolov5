from pathlib import Path
import os


img_path = os.path.abspath('F:\images')
def imgRename(dir):
    list_dir = os.listdir(dir)
    for idx,item in enumerate(list_dir):
        name_item_arr = item.split('.')
        target_name = 'b_'+str(idx)+'.'+name_item_arr[len(name_item_arr)-1]
        os.rename(os.path.join(dir,item),os.path.join(dir,target_name))
        print('deal:',idx+1)
   

imgRename(img_path)