#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/ImnetAggregate.c"
#else

static int nn_(ImnetAggregate_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
/*  long length = luaT_getfieldcheckint(L, 1, "length");*/
  long numClasses = luaT_getfieldcheckint(L, 1, "numclasses");
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  THTensor *agg = luaT_getfieldcheckudata(L, 1, "aggtensor", torch_Tensor);
  THTensor *chosen = luaT_getfieldcheckudata(L, 1, "chosentensor", torch_Tensor);

  THTensor_(resizeAs)(output, input);

  real * aggArray = THTensor_(data)(agg);
  real * chosenParent = THTensor_(data)(chosen);
  real * inputs=THTensor_(data)(input);
  real * outputs=THTensor_(data)(output);

 
 
		long iter=0;
		long k, cls, numparents, maxparent;
		real outvalue, parentval;
		long parents[3];
		real parentvalues[3];

		for (cls=0;cls<numClasses;cls++)
		{
			numparents=aggArray[iter];
			iter++;
			outvalue=inputs[cls];
			maxparent=-1;
			if (numparents>0) {
				/*/ get score values from parents + indices*/
				for (k=0;k<numparents;k++)
				{
					parents[k]=aggArray[iter+k];
					parentvalues[k]=inputs[parents[k]];
				}

				/*/ find the max one*/
				maxparent=parents[0];
				parentval=parentvalues[0];
				for (k=1;k<numparents;k++)
				{
					if (parentvalues[k] > parentval)
					{
						parentval=parentvalues[k];
						maxparent=parents[k];
					/*	printf("%d\n",cls);
						printf("%d\n",parents[k]);*/
					}
				}
				outvalue+=parentval;
			}
			chosenParent[cls]=maxparent;
			outputs[cls]=outvalue;
			iter=iter+numparents;
		
		}
  


  return 1;
}

static int nn_(ImnetAggregate_updateGradInput)(lua_State *L)
{
  long numClasses = luaT_getfieldcheckint(L, 1, "numclasses");
  THTensor *chosen = luaT_getfieldcheckudata(L, 1, "chosentensor", torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *gradInput  = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  THTensor_(resizeAs)(gradInput, gradOutput);
  THTensor_(copy)(gradInput, gradOutput);
  real * chosenParent = THTensor_(data)(chosen);
  real * gradOutputs = THTensor_(data)(gradOutput);
  real * gradInputs = THTensor_(data)(gradInput);

		long k, parent;

		for (k=numClasses-1; k>=0; k--) {
			parent=chosenParent[k];
			if (parent>-1) {
				gradInputs[parent] += gradOutputs[k];
			}
		}

  return 1;
}

static const struct luaL_Reg nn_(ImnetAggregate__) [] = {
  {"ImnetAggregate_updateOutput", nn_(ImnetAggregate_updateOutput)},
  {"ImnetAggregate_updateGradInput", nn_(ImnetAggregate_updateGradInput)},
  {NULL, NULL}
};

static void nn_(ImnetAggregate_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(ImnetAggregate__), "nn");
  lua_pop(L,1);
}

#endif
