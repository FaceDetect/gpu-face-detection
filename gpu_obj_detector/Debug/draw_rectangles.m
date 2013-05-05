#!/usr/bin/octave -qf
printf("Hello, world.");

args = argv();

imshow(imread(args{1}));

for i = 2:4:nargin
	%[str2num(args{i}), str2num(args{i+1}), str2num(args{i+2}), str2num(args{i+3})]
	%[args{i}, args{i+1}, args{i+2}, args{i+3}]
	%printf("\n");
	rectangle('Position', [str2num(args{i}), str2num(args{i+1}), str2num(args{i+2}), str2num(args{i+3})], 'EdgeColor', 'r');

end

pause;
