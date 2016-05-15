local cjson  = require 'cjson'
local torch =  require 'torch'
local f = io.open("split.json", "r")
local c = f:read "*a"
f:close()
content = cjson.decode(c)
torch.save('kaggle.t7', content)

local f = io.open("test.json", "r")
local b = f:read "*a"
f:close()

content = cjson.decode(b)
torch.save('kaggle_test.t7', content)