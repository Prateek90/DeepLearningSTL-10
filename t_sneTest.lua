m=require 'manifold'
N=1000

testset=

x=torch.FloatTensor(testset.data:size()):copy(testset.data)
x:resize(x:size(1),x:size(2)*x:size(3)*x:size(3))
labels=testset.labels

opts = {ndims = 2, perplexity = 30, pca = 50, use_bh = true, theta=0.5}
mapped_x1 = m.embedding.tsne(x, opts)

im_size = 4096
map_im = m.draw_image_map(mapped_x1, x:resize(x:size(1), 1, 28, 28), im_size, 0, true)
