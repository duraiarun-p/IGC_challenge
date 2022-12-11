clc;clear;close all;
%%
T=Tiff('/home/arun/Documents/PyWSPrecision/IGC/OS-3-cropped.tif','r');
I=read(T);

siz=size(I);
[I_1,I_2,I_3,I_4,mask_2_F,mask_2]=tumour_mask(I);

%%
final_mask=false(siz(1),siz(1));
tic;
for i=1:500:siz(1)
    for j=1:500:siz(2)

tileM=mask_2(i:i+499,j:j+499,:);
tile1=I_1(i:i+499,j:j+499,:);
tile2=I_2(i:i+499,j:j+499,:);
tile3=I_3(i:i+499,j:j+499,:);
tile4=I_4(i:i+499,j:j+499,:);

tile_V=histeq(tile2(:,:,3));
tile_V(tile_V>0.7)=0;
tile_V(tile_V<0.4)=0;

tile_Vm=imbinarize(tile_V,'global');
tile_Vm_s=immultiply(tileM,imfill(tile_Vm,'holes'));
final_mask(i:i+499,j:j+499,:)=tile_Vm_s;
    end
end
time=toc;

mask_F=zeros(size(I));
mask_F(:,:,1)=final_mask;
mask_F(:,:,2)=final_mask;
mask_F(:,:,3)=final_mask;
% % I_1=immultiply(I_1,(mask_2_F));
I_1_F=I_1.*uint8(mask_F);

I_1_FM=imoverlay(I,final_mask,'g');

figure(9),imshow(final_mask);
figure(10),imshow(I_1_F);
figure(11),imshow(I_1_FM);
%%
function [I_1,I_2,I_3,I_4,mask_2_F,mask_2]=tumour_mask(I)
R=rescale(I(:,:,1));
Rm=imbinarize(R,0.6);

G=rescale(I(:,:,2));
Gm=imbinarize(G,0.6);

B=rescale(I(:,:,3));
Bm=imbinarize(B,0.6);
%%
I1 = rgb2hsv(I);
H=I1(:,:,1);
Hm=imbinarize(H, 0.55);
%%
I2=rgb2lab(I);
% L=I2(:,:,1);
% A=rescale(I2(:,:,2));
B1=rescale(I2(:,:,3));
Bm1=imbinarize(B1, 0.65);
%%
I3=rgb2ycbcr(I);
%%
mask1=imcomplement(Hm);
mask2=Bm1;
mask3=imcomplement(Bm);
mask4=imcomplement(Rm);
mask5=imcomplement(Gm);
mask12=immultiply(mask1,mask2);
mask23=immultiply(mask2,mask3);
mask45=immultiply(mask4,mask5);

mask_A=immultiply(mask12,mask23);
mask=immultiply(mask_A,mask45);

%%
SE = strel("diamond",15);

mask_1=imclose(mask,SE);

SE = strel("diamond",5);
mask_2=imfill(mask_1,8);
%%

mask_2_F=zeros(size(I));
mask_2_F(:,:,1)=mask_2;
mask_2_F(:,:,2)=mask_2;
mask_2_F(:,:,3)=mask_2;

I_1=I.*uint8(mask_2_F);
I_2=I1.*(mask_2_F);
I_3=I2.*(mask_2_F);
I_4=I3.*uint8(mask_2_F);
%%
end