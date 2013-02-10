function  test_suite = test_save_haar_cascade
initTestSuite;

function testMatFile = setup
testMatFile = 'test-save.mat';

function test_save_cascade(testMatFile)
get_haar_cascade('test-haarcascade.xml', testMatFile, 1);
assertEqual(exist(testMatFile), 2);


function test_implicit_save_cascade(testMatFile)

if exist(testMatFile)
    delete(testMatFile)
end

get_haar_cascade('test-haarcascade.xml', testMatFile);
assertEqual(exist(testMatFile), 2);

function test_not_save(testMatFile)

if exist(testMatFile)
    delete(testMatFile)
end

get_haar_cascade('test-haarcascade.xml');
assertEqual(exist(testMatFile), 0);