with open(r"C:\Users\yashr\Desktop\NEU\Semester 4\DL\Project\flickr8k\captions.txt") as f:
    for i, line in enumerate(f):
        print(repr(line))
        if i >= 3: break