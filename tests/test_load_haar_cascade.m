function test_suite = test_load_haar_cascade
initTestSuite;

function haarCascade = setup
toReload = true;
matFileName = 'test-haarcascade.mat';
testFile = dir('test_load_haar_cascade.m');
funcFile = dir('../get_haar_cascade.m');

if exist(matFileName)
    matFile = dir(matFileName);
    if (testFile.datenum < matFile.datenum) && (funcFile.datenum < matFile.datenum)
        toReload = false;
    end
end

haarCascade = get_haar_cascade('test-haarcascade.xml', 'test-haarcascade.mat', toReload);

function test_get_size(haarCascade)
assertEqual(haarCascade.size.w, 20);
assertEqual(haarCascade.size.h, 15);

function test_get_stages(haarCascade)
assertEqual(length(haarCascade.stages), 4);

function test_get_stage_threshold(haarCascade)
assertElementsAlmostEqual(haarCascade.stages(1).threshold, 0.8226894140243530);
assertElementsAlmostEqual(haarCascade.stages(2).threshold, 6.9566087722778320);
assertElementsAlmostEqual(haarCascade.stages(4).threshold, 18.4129695892333980);

function test_get_stages_parent(haarCascade)
assertEqual(haarCascade.stages(1).parent, 0);
assertEqual(haarCascade.stages(2).parent, 1);
assertEqual(haarCascade.stages(3).parent, 2);
assertEqual(haarCascade.stages(4).parent, 3);

function test_get_stage_trees(haarCascade)
assertEqual(length(haarCascade.stages(1).trees), 3);
assertEqual(length(haarCascade.stages(2).trees), 16);
assertEqual(length(haarCascade.stages(3).trees), 21);
assertEqual(length(haarCascade.stages(4).trees), 39);


function test_get_tree_threshold(haarCascade)
assertElementsAlmostEqual(haarCascade.stages(1).trees(2).threshold, 0.0151513395830989);
assertElementsAlmostEqual(haarCascade.stages(2).trees(6).threshold, 0.0366676896810532);
assertElementsAlmostEqual(haarCascade.stages(3).trees(20).threshold, 4.5017572119832039e-003);
assertElementsAlmostEqual(haarCascade.stages(4).trees(14).threshold, 0.0240177996456623);

function test_get_tree_leftval(haarCascade)
assertElementsAlmostEqual(haarCascade.stages(1).trees(2).leftVal, 0.1514132022857666);
assertElementsAlmostEqual(haarCascade.stages(2).trees(6).leftVal, 0.3675672113895416);
assertElementsAlmostEqual(haarCascade.stages(3).trees(20).leftVal, 0.4509715139865875);
assertElementsAlmostEqual(haarCascade.stages(4).trees(14).leftVal, 0.5797107815742493);

function test_get_tree_rightval(haarCascade)
assertElementsAlmostEqual(haarCascade.stages(1).trees(2).rightVal, 0.7488812208175659);
assertElementsAlmostEqual(haarCascade.stages(2).trees(6).rightVal, 0.7920318245887756);
assertElementsAlmostEqual(haarCascade.stages(3).trees(20).rightVal, 0.7801457047462463);
assertElementsAlmostEqual(haarCascade.stages(4).trees(14).rightVal, 0.2751705944538117);

function test_get_tree_rects(haarCascade)
assertEqual(haarCascade.stages(3).trees(15).feature.rects(3).x, 5);
assertEqual(haarCascade.stages(3).trees(15).feature.rects(3).y, 5);
assertEqual(haarCascade.stages(3).trees(15).feature.rects(3).w, 2);
assertEqual(haarCascade.stages(3).trees(15).feature.rects(3).h, 4);
assertEqual(haarCascade.stages(3).trees(15).feature.rects(3).wg, 2);

