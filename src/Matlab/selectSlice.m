function  selectSlice(inputImage,slice)

%img_uint8 = uint8(inputImage(:,:,slice));
img_uint8 = inputImage(:,:,slice);
img_uint8 = imrotate(img_uint8,90);
ax1 = axes;
imagesc (img_uint8);
colormap(ax1,'gray');
set(gca,'XColor', 'none','YColor','none')
end

