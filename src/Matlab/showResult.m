function showResult(slice)

CPUTh = 'D:\Users\ealcain\PhD_Edu\Code\VS_C++\Intel\NewPatchedBrainPureC\x64\ReleaseIntelMatlab\Subject_08_AtlasSize_01_7Nh_02Th_ICCTh.mat';
CPUNew='D:\Users\ealcain\PhD_Edu\Code\VS_C++\Intel\NewPatchedBrainPureC\x64\ReleaseIntelMatlab\Subject_08_AtlasSize_01_7Nh_02Th_ICC.mat';
GPU= 'D:\Users\ealcain\PhD_Edu\Code\VS_C++\CUDA\PatchBasedPseudoCT\x64\Release_MATLAB\Subject_08_AtlasSize_01_Op2_Rd_7Nh.mat';

load(GPU);
load(CPUTh);
 load(CPUNew);
 close all;
 figure;
 selectSlice(Subject_08_AtlasSize_01_Op2_Rd_7Nh, slice);
 title('Subject_01_AtlasSize_01_Op2_Rd_7Nh');
 figure;
 selectSlice(Subject_08_AtlasSize_01_7Nh_02Th_ICC, slice);
  title('Subject_08_AtlasSize_01_7Nh_02Th_ICC');
 figure;
 selectSlice(Subject_08_AtlasSize_01_7Nh_02Th_ICCTh, slice);
  title('Subject_08_AtlasSize_01_7Nh_02Th_ICCTh');
  
  
  comparePatchBasedCTSingle(GPU,CPUNew);
  %comparePatchBasedCTSingle(GPU,CPUTh);
end








