local testSgemm, parent = torch.class('nxn.testSgemm', 'nxn.Module')

function testSgemm:__init(m,n,k,lda,ldb,ldc, tA, tB)
   parent.__init(self)

	self.m=m
	self.n=n
	self.k=k
	self.lda=lda
	self.ldb=ldb
	self.ldc=ldc

	self.tA=tA or 0
	self.tB=tB or 0

end


function testSgemm:run(A,B,C)
   A.nxn.testSgemm_run(self,A,B,C)
   return C
end

