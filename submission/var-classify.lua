--
--  Copyright (c) 2016, Manuel Araoz
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  classifies an image using a trained model
--

require 'torch'
require 'paths'
require 'cudnn'
require 'cunn'
require 'image'

local t = require '../datasets/transforms'
-- local imagenetLabel = require './imagenet'

if #arg < 2 then
   io.stderr:write('Usage: th classify.lua [MODEL] [FILE]...\n')
   os.exit(1)
end
for _, f in ipairs(arg) do
   if not paths.filep(f) then
      io.stderr:write('file not found: ' .. f .. '\n')
      os.exit(1)
   end
end


-- Load the model
local model = torch.load(arg[1])
local softMaxLayer = cudnn.SoftMax():cuda()
local testdata = torch.load(arg[2])

-- add Softmax layer
model:add(softMaxLayer)

-- Evaluate mode
model:evaluate()

-- The model was trained with this input normalization
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

local transform = t.Compose{
   t.Scale(256),
   t.ColorNormalize(meanstd),
   -- t.CenterCrop(224),
   t.TenCrop(224)
}

local features
local N = 5

-- for i=2,#testdata['image'] do
file = io.open("submission.csv", "w")
file:write("c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,img\n")



for i=1,#testdata['name'] do
   -- load the image as a RGB float tensor with values 0..1
   local img = image.load(testdata['image'][i], 3, 'float')
   local name = testdata['name'][i]
   -- print(name)
   if i%100 ==0 then
      print(i)
   end
   -- Scale, normalize, and crop the image
   batch = transform(img)
   -- View as mini-batch of size 1
   -- batch = img:view(1, table.unpack(img:size():totable()))

   -- Get the output of the softmax
   local output = model:forward(batch:cuda()):squeeze()
   local mu = torch.mean(output,1)
   for i=1,mu:size(2) do
      
      
      file:write(mu[1][i] ..' ,' )
   end
   file:write(name ..'\n')
end

file:close()
