-- tostring() functions for Tensor and Storage
local function Storage__printformat(self)
   local intMode = true
   local type = torch.typename(self)
--   if type == 'torch.FloatStorage' or type == 'torch.DoubleStorage' then
      for i=1,self:size() do
         if self[i] ~= math.ceil(self[i]) then
            intMode = false
            break
         end
      end
--   end
   local tensor = torch.DoubleTensor(torch.DoubleStorage(self:size()):copy(self), 1, self:size()):abs()
   local expMin = tensor:min()
   if expMin ~= 0 then
      expMin = math.floor(math.log10(expMin)) + 1
   end
   local expMax = tensor:max()
   if expMax ~= 0 then
      expMax = math.floor(math.log10(expMax)) + 1
   end

   local format
   local scale
   local sz
   if intMode then
      if expMax > 9 then
         format = "%11.4e"
         sz = 11
      else
         format = "%SZd"
         sz = expMax + 1
      end
   else
      if expMax-expMin > 4 then
         format = "%SZ.4e"
         sz = 11
         if math.abs(expMax) > 99 or math.abs(expMin) > 99 then
            sz = sz + 1
         end
      else
         if expMax > 5 or expMax < 0 then
            format = "%SZ.4f"
            sz = 7
            scale = math.pow(10, expMax-1)
         else
            format = "%SZ.4f"
            if expMax == 0 then
               sz = 7
            else
               sz = expMax+6
            end
         end
      end
   end
   format = string.gsub(format, 'SZ', sz)
   if scale == 1 then
      scale = nil
   end
   return format, scale, sz
end

local function Storage__tostring(self)
   local strt = {'\n'}
   local format,scale = Storage__printformat(self)
   if scale then
      table.insert(strt, string.format('%g', scale) .. ' *\n')
      for i = 1,self:size() do
         table.insert(strt, string.format(format, self[i]/scale) .. '\n')
      end
   else
      for i = 1,self:size() do
         table.insert(strt, string.format(format, self[i]) .. '\n')
      end
   end
   table.insert(strt, '[' .. torch.typename(self) .. ' of size ' .. self:size() .. ']\n')
   str = table.concat(strt)
   return str
end

rawset(torch.getmetatable('torch.ByteStorage'), '__tostring__', Storage__tostring)
rawset(torch.getmetatable('torch.CharStorage'), '__tostring__', Storage__tostring)
rawset(torch.getmetatable('torch.ShortStorage'), '__tostring__', Storage__tostring)
rawset(torch.getmetatable('torch.IntStorage'), '__tostring__', Storage__tostring)
rawset(torch.getmetatable('torch.LongStorage'), '__tostring__', Storage__tostring)
rawset(torch.getmetatable('torch.FloatStorage'), '__tostring__', Storage__tostring)
rawset(torch.getmetatable('torch.DoubleStorage'), '__tostring__', Storage__tostring)

local function Tensor__printMatrix(self, indent)
   local format,scale,sz = Storage__printformat(self:storage())
--   print('format = ' .. format)
   scale = scale or 1
   indent = indent or ''
   local strt = {indent}
   local nColumnPerLine = math.floor((80-#indent)/(sz+1))
--   print('sz = ' .. sz .. ' and nColumnPerLine = ' .. nColumnPerLine)
   local firstColumn = 1
   local lastColumn = -1
   while firstColumn <= self:size(2) do
      if firstColumn + nColumnPerLine - 1 <= self:size(2) then
         lastColumn = firstColumn + nColumnPerLine - 1
      else
         lastColumn = self:size(2)
      end
      if nColumnPerLine < self:size(2) then
         if firstColumn ~= 1 then
            table.insert(strt, '\n')
         end
         table.insert(strt, 'Columns ' .. firstColumn .. ' to ' .. lastColumn .. '\n' .. indent)
      end
      if scale ~= 1 then
         table.insert(strt, string.format('%g', scale) .. ' *\n ' .. indent)
      end
      for l=1,self:size(1) do
         local row = self:select(1, l)
         for c=firstColumn,lastColumn do
            table.insert(strt, string.format(format, row[c]/scale))
            if c == lastColumn then
               table.insert(strt, '\n')
               if l~=self:size(1) then
                  if scale ~= 1 then
                     table.insert(strt, indent .. ' ')
                  else
                     table.insert(strt, indent)
                  end
               end
            else
               table.insert(strt, ' ')
            end
         end
      end
      firstColumn = lastColumn + 1
   end
   local str = table.concat(strt)
   return str
end

local function Tensor__printTensor(self)
   local counter = torch.LongStorage(self:nDimension()-2)
   local strt = {''}
   local finished
   counter:fill(1)
   counter[1] = 0
   while true do
      for i=1,self:nDimension()-2 do
         counter[i] = counter[i] + 1
         if counter[i] > self:size(i) then
            if i == self:nDimension()-2 then
               finished = true
               break
            end
            counter[i] = 1
         else
            break
         end
      end
      if finished then
         break
      end
--      print(counter)
      if #strt > 1 then
         table.insert(strt, '\n')
      end
      table.insert(strt, '(')
      local tensor = self
      for i=1,self:nDimension()-2 do
         tensor = tensor:select(1, counter[i])
         table.insert(strt, counter[i] .. ',')
      end
      table.insert(strt, '.,.) = \n')
      table.insert(strt, Tensor__printMatrix(tensor, ' '))
   end
   local str = table.concat(strt)
   return str
end

local function Tensor__tostring(self)
   local str = '\n'
   local strt = {''}
   if self:nDimension() == 0 then
      table.insert(strt, '[' .. torch.typename(self) .. ' with no dimension]\n')
   else
      local tensor = torch.DoubleTensor():resize(self:size()):copy(self)
      if tensor:nDimension() == 1 then
         local format,scale,sz = Storage__printformat(tensor:storage())
         if scale then
            table.insert(strt, string.format('%g', scale) .. ' *\n')
            for i = 1,tensor:size(1) do
               table.insert(strt, string.format(format, tensor[i]/scale) .. '\n')
            end
         else
            for i = 1,tensor:size(1) do
               table.insert(strt, string.format(format, tensor[i]) .. '\n')
            end
         end
         table.insert(strt, '[' .. torch.typename(self) .. ' of dimension ' .. tensor:size(1) .. ']\n')
      elseif tensor:nDimension() == 2 then
         table.insert(strt, Tensor__printMatrix(tensor))
         table.insert(strt, '[' .. torch.typename(self) .. ' of dimension ' .. tensor:size(1) .. 'x' .. tensor:size(2) .. ']\n')
      else
         table.insert(strt, Tensor__printTensor(tensor))
         table.insert(strt, '[' .. torch.typename(self) .. ' of dimension ')
         for i=1,tensor:nDimension() do
            table.insert(strt, tensor:size(i))
            if i ~= tensor:nDimension() then
               table.insert(strt, 'x')
            end
         end
         table.insert(strt, ']\n')
      end
   end
   local str = table.concat(strt)
   return str
end
rawset(torch.getmetatable('torch.ByteTensor'), '__tostring__', Tensor__tostring)
rawset(torch.getmetatable('torch.CharTensor'), '__tostring__', Tensor__tostring)
rawset(torch.getmetatable('torch.ShortTensor'), '__tostring__', Tensor__tostring)
rawset(torch.getmetatable('torch.IntTensor'), '__tostring__', Tensor__tostring)
rawset(torch.getmetatable('torch.LongTensor'), '__tostring__', Tensor__tostring)
rawset(torch.getmetatable('torch.FloatTensor'), '__tostring__', Tensor__tostring)
rawset(torch.getmetatable('torch.DoubleTensor'), '__tostring__', Tensor__tostring)

local function Tensor__type(self,type)
   local current = torch.typename(self)
   if not type then return current end
   if type ~= current then
      local new = torch.getmetatable(type).new()
      if self:nElement() > 0 then
         new:resize(self:size()):copy(self)
      end
      return new
   else
      return self
   end
end
local function Tensor__typeAs(self,tensor)
   return self:type(tensor:type())
end
local function Tensor__double(self,type)
   return self:type('torch.DoubleTensor')
end
local function Tensor__float(self,type)
   return self:type('torch.FloatTensor')
end
rawset(torch.getmetatable('torch.DoubleTensor'), 'type', Tensor__type)
rawset(torch.getmetatable('torch.DoubleTensor'), 'typeAs', Tensor__typeAs)
rawset(torch.getmetatable('torch.DoubleTensor'), 'double', Tensor__double)
rawset(torch.getmetatable('torch.DoubleTensor'), 'float', Tensor__float)
rawset(torch.getmetatable('torch.FloatTensor'), 'type', Tensor__type)
rawset(torch.getmetatable('torch.FloatTensor'), 'typeAs', Tensor__typeAs)
rawset(torch.getmetatable('torch.FloatTensor'), 'double', Tensor__double)
rawset(torch.getmetatable('torch.FloatTensor'), 'float', Tensor__float)

rawset(torch.getmetatable('torch.ByteTensor'), 'type', Tensor__type)
rawset(torch.getmetatable('torch.CharTensor'), 'type', Tensor__type)
rawset(torch.getmetatable('torch.ShortTensor'), 'type', Tensor__type)
rawset(torch.getmetatable('torch.IntTensor'), 'type', Tensor__type)
rawset(torch.getmetatable('torch.LongTensor'), 'type', Tensor__type)

rawset(torch.getmetatable('torch.ByteTensor'), 'typeAs', Tensor__typeAs)
rawset(torch.getmetatable('torch.CharTensor'), 'typeAs', Tensor__typeAs)
rawset(torch.getmetatable('torch.ShortTensor'), 'typeAs', Tensor__typeAs)
rawset(torch.getmetatable('torch.IntTensor'), 'typeAs', Tensor__typeAs)
rawset(torch.getmetatable('torch.LongTensor'), 'typeAs', Tensor__typeAs)
