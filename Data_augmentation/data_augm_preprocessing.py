import os
import cv2 as cv

## TU Graz:
#w = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2291166109949928, 0.0, 0.0, 0.0, 0.0, 0.05025179610212472, 0.026120556917357606, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001985722330912488, 0.0, 0.0, 0.0, 0.06457682904199318, 0.0, 0.0, 0.0, 0.14188562294703053, 0.0, 0.0636973569845078, 0.0, 0.06233845713132418, 0.0, 0.00859711714996289, 0.0, 0.024766028092905158, 0.0009700773457642126, 0.0, 0.0, 0.16997551732670044, 0.13554935592977568, 0.0, 0.05322643124770099, 0.2819035341302706, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06626431505823681, 0.0, 0.0, 0.0, 0.0, 0.009837559816400273, 0.06602575789144703, 0.259749054232675, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.008274874265531666, 0.057060918365353944, 0.0, 0.0, 0.002301439783332628, 0.0, 0.0, 0.0, 0.0, 0.0, 0.028682854417667415, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.005559968228752979, 0.10947581564023934, 0.0, 0.026497568343543904, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03383661803100442, 0.0, 0.0, 0.0, 0.0, 0.0, 0.20362195701571056, 0.0, 0.0, 0.006832749017731293, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6015555309576054, 0.06904342560609958, 0.0, 0.03431662966555772, 0.006364201610094177, 0.0, 0.01202203225407408, 0.0, 0.04837008535322956, 0.0, 0.10812675987564901, 0.0, 0.0, 0.0, 0.23118045019762073, 0.0, 0.0047722286242569845, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0014030084929214804, 0.0, 0.16883801785260855, 0.0637127230108087, 0.047289796901601934, 0.0, 0.0, 0.0, 0.0, 0.03806460434558962, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1691721735733585, 0.0, 0.04510754385907799, 0.0, 0.0, 0.21485009587743314, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.21425246195165623, 0.0, 0.0022168040911591507, 0.0, 0.0, 0.0, 0.0261136212711103, 0.0024154353480407955, 0.16743548591872545, 0.0, 0.011286137940239135, 0.0, 0.004430497185771626, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02176807414231444, 0.0, 0.0, 0.0, 0.13243672637535078, 0.0, 0.018859754787376042, 0.0, 0.038615795299173125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.024275827044905558, 0.0, 0.15843888655592123, 0.0, 0.0, 0.00485683407336033, 0.0, 0.06894059741139093, 0.0, 0.0, 0.029948794072743017, 0.0, 0.0, 0.38147330585454475, 0.01406304933786476, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0579070895413071, 0.0, 0.0, 0.02570053907653845, 0.005455800205731846, 0.0, 0.0, 0.0, 0.05012516642740184, 0.11526778941474071, 0.0, 0.0, 0.0, 0.0, 0.04278417802488989, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09924869302534521, 0.0, 0.0, 0.0018522798667921052, 0.0, 0.0, 0.043669757024922526, 0.0, 0.0, 0.10565707272449076, 0.0, 0.0, 0.17661172441408857, 0.09083895413943072, 0.0, 0.0, 0.0, 0.0, 0.10481991249544247, 0.0, 0.0, 0.0, 0.0, 0.019906246948142844, 0.0, 0.0, 0.0, 0.017933526481981835, 0.0, 0.027897771108065614, 0.0, 0.0, 0.0, 0.02346120291157207, 0.060932378016426154, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.029926786313044667, 0.0, 0.0, 0.17424227626395808, 0.0, 0.0, 0.0, 0.0, 0.0, 0.061110423895988905, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35934112438555943, 0.0, 0.09429638238099423, 0.0, 0.0, 0.02055365304235732, 0.015609079633976035, 0.0, 0.0, 0.0, 0.0, 0.05923005807442902, 0.06324572626616791, 0.0, 0.0, 0.0, 0.17304605727940722, 0.0, 0.0, 0.0, 0.06582069971320027, 0.005289851298210891, 0.0, 0.0, 0.10152508992659386, 0.0, 0.012822349103835312, 0.0, 0.0, 0.0, 0.0, 0.10375405289269111, 0.0, 0.0, 0.0, 0.0, 0.01804716873647032, 0.0, 0.0, 0.0, 0.028343560279301423, 0.19419302746558662, 0.0, 0.0, 0.0, 0.0, 0.12316191431110649, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.022136425488140734, 0.0, 0.0, 0.0, 0.0, 0.18240478182593606, 0.0, 0.0, 0.0, 0.0, 0.0019921939850064616, 0.0187572431214761, 0.0, 0.07226938903067869, 0.0, 0.0, 0.0, 0.004042981910748407, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10864470009179866, 0.0, 0.0, 0.05005615325770276, 0.0, 0.0, 0.0, 0.16404334903525936, 0.0, 0.0, 0.06091836535394111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
## drone deploy (DD):
w = [0.05066750762115001, 0.04460652185402463, 0.2008417239358585, 0.039643105198482666, 0.0, 0.035882661079099, 0.11390053606496471, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.09690601284296556, 0.0003650015642924184, 0.009821335283669418, 0.019654992158912702, 0.015843965697552812, 0.145436955820703, 0.12168771835528083, 0.030722289441843336, 1.0, 0.007772966769262872, 0.004747000521648409, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9033157894736842, 0.23020635923352284, 0.1934856842658261, 0.9537405699916177, 0.002502737368997341, 0.00031269543464665416, 0.05168765210030685, 0.41824894514767935, 0.9724094685687412, 0.005948964149663414, 0.9901264235712047, 0.12542176296921131, 0.016567054629338367, 0.0, 0.0, 0.1825446898002103, 0.0, 0.0, 0.0, 0.0, 0.022715349910817334, 0.04298024379991593, 0.8936249467858663, 0.9776076049643517, 0.11122265666279192, 0.46770094249708777, 0.0, 0.06415789473684211, 0.21174367870472585, 0.038061379202174936, 0.0845262440354465, 0.0, 0.0, 0.02388106911276293, 0.0, 0.040212363330529854, 0.21868755915448523, 0.0048051812389010755, 0.0, 0.0, 0.9239381833473507, 0.9650159493803273, 0.9684689395523949, 0.6719220835126991, 0.10116321911679563, 0.2156327152387456, 0.7369626591381239, 0.0, 0.0585069902400422, 0.9868262846986251, 1.0, 0.05109795084115088, 0.4659348778299646, 0.0, 0.6046960437439691, 0.700591330765475, 0.0, 0.0, 0.04711270200557983, 0.0, 0.02319301032565528, 0.003912771285475793, 0.0, 0.013835027774866367, 0.0, 0.05028109073714076, 0.0014084507042253522, 0.0, 0.0062131258810630184, 0.016739028090181515, 0.0019324176111140126, 0.0304696089209989, 0.2156327152387456, 0.71734395750332, 0.0, 5.212133847597206e-05, 0.0003127117319018085, 0.023603461841070025, 0.0, 0.01165891148638051, 0.0, 0.42172810816520545, 0.0, 0.0, 0.15066225165562913, 0.004965243296921549, 0.025321120235839125, 0.0, 0.006017791732077446, 0.012979536295598471, 0.0, 0.6013773222293401, 0.027470491716508758, 0.0, 0.0, 0.0, 0.0, 0.39707291112293774, 0.30818931554341544, 0.5169369080109166, 0.2763546533483701, 0.0563275697464729, 0.015025391340767498, 0.0, 0.046805584937592556, 0.0, 0.692795819335573, 0.01887385970430953, 0.26178594181354853, 0.1125900983848056, 0.9789695057833859, 0.2506813445198525, 0.11290663286861005, 0.5778769051032872, 0.027665889336442654, 0.0923270836615724, 0.0, 0.04345319215850718, 0.0326522013903518, 0.08497940213372769, 0.7221574652961746, 0.9297498949138293, 0.6889952153110048, 0.0026093309675399227, 0.0021935551261294197, 0.03518890887924648, 0.0066290844555799144, 0.5391616253546764, 0.864648591734667, 0.723035027765912, 0.0, 0.5743633640059919, 0.031066021867115223, 0.0, 0.0, 0.00266096212042158, 0.12053971433089126, 0.4988828598787105, 0.14326571685821726, 0.0329388353581902, 0.006944444444444444, 0.29646958740961293, 0.03299012397562513, 0.04114189756507137, 0.007740585774058577, 0.0, 0.012720105124835743, 0.0, 0.0, 0.248000423751258, 0.0, 0.0, 0.22316847394869083, 0.12537756345715648, 0.0016697975370486328, 0.0, 0.0, 0.09948092256649008, 0.10195718984202297, 0.03200842326928139, 0.0, 0.007054116656138134, 0.04971098265895954, 0.01049317943336831, 0.003390714658320292, 0.008901921767816935, 0.0, 0.0008342892898112421, 0.34529932658147305, 0.0, 0.4788505870369378, 0.8820340053692688, 0.03546666666666667, 0.0, 0.007234221010694066, 0.022153556090918614, 0.0, 0.0, 0.0, 0.007903690133472913, 0.008414340963729486, 0.0, 0.6495805142083897, 0.0, 0.0, 0.005374099968694563, 0.021416890858258163, 0.0, 0.049705585910561775, 0.01243144424131627, 0.0, 0.015872181636621643, 0.6933679690399194, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0006778247041034465, 0.07712990299451708, 0.0, 0.4659454274141974, 0.005752536345570547, 0.0, 0.03144026873819022, 0.4866080994214699, 0.0, 0.0, 0.5161481003815358, 0.0, 0.0, 0.011197739521741405, 0.0, 0.0, 0.0, 0.3534833288153971, 0.0, 0.04295323340924739, 0.08068655758742381, 0.0, 0.35647685647685645, 0.0, 0.0748947957172535, 0.1041411288173533, 0.00015633956954505185, 0.02359403609730578, 0.12836558687421118, 0.0, 0.4328318252395995, 0.0, 0.0, 0.0, 0.0038656427937104948, 0.009382043084019078, 0.13285190888334306, 0.0, 0.008872651356993737, 0.003023983315954119, 0.04901598691500026, 0.0539911073470252, 0.010876954452753228, 0.0, 0.000834680995357087, 0.5281903849225363, 0.0052072375341889335, 0.0, 0.0, 0.0, 0.34450189953566906, 0.2918498120002118, 0.0, 0.02972119936875329, 0.0, 0.0, 0.026041666666666668, 0.0, 0.0, 0.0, 0.3430906468671545, 0.005486100121327215, 0.0, 0.0, 0.18551624471444628, 0.2674648564861831, 0.0027199497855424208, 0.0, 0.019209515288669016, 0.45886244797780384, 0.3625578231292517, 0.15853069115514742, 0.0, 0.2067928906703233, 0.0004692632566870014, 0.08577998823466496, 0.1612783940834654, 0.0, 0.0, 0.018976724680226462, 0.6445590894585682, 0.2342480211081794, 0.0, 0.0, 0.35420402996096634, 0.012960543506663183, 0.006692460524939873, 0.0753899021940259, 0.12049100893722407, 0.004353070750511355, 0.0026659696811291165, 0.0, 0.009579145728643216, 0.16882157369782047, 0.4472305983177273, 0.02413122338468009, 0.0, 0.0, 0.016458092333578714, 0.0, 0.0, 0.013443421191912776, 0.05442794573229356, 0.3836668469442942, 0.02546600157521659, 0.13156490179333902, 0.7825459165992839, 0.052843825285429975, 0.0, 0.8488138990978951, 0.5983941029353692, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.997073272708268, 0.00998014421569652, 0.09146049481245012, 0.0581547650233143, 0.0, 0.0, 0.0, 0.4647360547125454, 0.13970626941095962, 0.0, 0.0, 0.05991296964551051, 0.0011986658328121743, 0.0, 0.08888888888888889, 0.0, 0.022739208066379583, 0.0010956331194240101, 0.0, 0.08970807585766034, 0.0033397693471794603, 0.04129452558579384, 0.1862807128545819, 0.0971022937966838, 0.0028726626971691217, 0.05765491144163557, 0.0, 0.04027799715684726, 0.007140606951590885, 0.0, 0.12010875359846465, 0.08257221750212404, 0.056110643668489696, 0.0, 0.2943237003423294, 0.12242959320518551, 0.0019814370633016998, 0.0, 0.08224563610630602, 0.0, 0.0, 0.00512498692605376, 0.0, 0.00015634771732332708, 0.07453284632893971, 0.00224285416232005, 0.016407599309153715, 0.12059782608695652, 0.02099236641221374, 0.03120902176763703, 0.32025416997617157, 0.021993522098004387, 0.0, 0.0014596257102642966, 0.22211991550668905, 0.3256740031211322, 0.04300567107750473, 0.038733315885893745, 0.3938338084679548, 0.00909090909090909, 0.08370774497218657, 0.007621235057681266, 0.07308018667798048, 0.15234828496042216, 0.09495089666951324, 0.0564744059457762, 0.030103480714957668, 0.16235663276347112, 0.003187875620590541, 0.046330420132673474, 0.01802037085400888, 0.035849453322119425, 0.0, 0.07445293962562616, 0.48737320371935755, 0.013157894736842105, 0.008369076263207448, 0.006837874517172982, 0.0, 0.0042357370705433245, 0.0077953332635764365, 0.0, 0.042516200410937253, 0.0, 0.04396831304162901, 0.05364186168814094, 0.0, 0.016891180804041256, 0.0, 0.0, 0.0, 0.03580698130197574, 0.0, 0.061765017480665325, 0.0, 0.0016177007775400511, 0.26231466898387007, 0.042045811107624724, 0.0, 0.013932464770007976, 0.002459445316588174, 0.0, 0.055253333333333335, 0.041734860883797055, 0.000835291046724093, 0.1960514476261049, 0.0, 0.030562411384760804, 0.01320062860136197, 0.008391901814748767, 0.017223272421760135, 0.012954318980437404, 0.25819176353021617, 0.0, 0.047556281963410135, 0.004387797743418303, 0.02396250263324205, 0.0, 0.0, 0.04001678292337547, 0.6813570069752695, 0.0, 0.2118182299105962, 0.159024862770423, 0.005813953488372093, 0.003081583620599603, 0.19418396375513644, 0.0, 0.01744671959882992, 0.5557078106883103, 0.0, 0.11653057913805223, 0.020192409345596785, 0.0, 0.011396309268649695, 0.01643835616438356, 0.10791904736476744, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00020847448793453902, 0.0, 0.0029300962745918793, 0.33410660621078714, 0.329973474801061, 0.33410660621078714, 0.0, 0.33410660621078714, 0.2163979929539874, 0.6798820618122466, 0.48240252897787145, 1.0, 1.0, 0.9999479030997656, 0.14783799441723283, 0.12791493393693742, 0.9579669179229481, 0.10502525923956395, 1.0, 0.6292939936775553, 0.0, 1.0, 0.9882802142440473, 0.0, 0.0860254656424287, 0.0, 0.00114726741760534, 0.0, 0.4374305518810519, 0.9655406818657185, 0.02622505485320238, 0.025116425095494742, 0.0, 0.0, 0.0, 0.00020846362309776944, 0.37959936742224565, 0.0, 0.15339028296849974, 0.0, 1.0, 0.15094946894110073, 0.003131033762980744, 0.0, 0.0, 0.12629454564767115, 0.1631235979062066, 0.132487554199454, 0.08082467670129319, 0.2041784362841763, 0.0, 0.2527350562866656, 0.0, 0.09101986754966887, 0.20397400383549968, 0.019910690832676648, 0.07371693539888291, 0.07599619490540112, 0.21124733475479743, 0.14824654622741765, 0.20138445154419596, 0.15609393263202634, 0.14267929634641408, 0.0, 0.0, 0.0, 0.18122130885021887, 0.14957401490947816, 0.0, 0.05536259341121987, 0.16463641683068247, 0.0, 0.18199284837487326, 0.9546550639547075]

img_nr = [i > 0.05 for i in w]
nr = []
for i in range(len(w)):
    if img_nr[i] == True:
        nr.append(i)

nr2 = []
for el in nr:
    el += 1
    nr2.append(str(el).zfill(4))
nr = nr2

path = "C:\\Users\\Simon\\Desktop\\plains"

def copy_image(path): # copy image if also in nr[] list
    os.chdir(path)
    files = os.listdir()
    j = 1
    for i in range(len(w)):
        if nr[j] == files[i][:4]:
            img = cv.imread(files[i])
            img_name = "_" + files[i]
            cv.imwrite(img_name, img)
            if j <= len(nr) - 2:
                j += 1

def copy_image2(path):
    os.chdir(path)
    files = os.listdir()
    # nr = [] # to be filled with file names from other folder with water images
    j = 0
    for i in range(len(files)):
        if nr[j] == files[i]:
            img = cv.imread(files[i])
            img_name = "_" + files[i]
            cv.imwrite(img_name, img)
            if j <= len(nr) - 2:
                j += 1

def rename_image(path):
    os.chdir(path)
    files = os.listdir()
    count = 24
    for file in files:
        img = cv.imread(file)
        end = file.find(".")
        img_name = str(count) + "_" + file[1:end] + file[-4:]
        cv.imwrite(img_name, img)
        count += 1

def rename_image2(path): # increase renaming every other time
    os.chdir(path)
    files = os.listdir()
    count = 24
    count2 = 0
    for file in files:
        img = cv.imread(file)
        splt = file.split("_")
        splt2 = str(count) + "_" + splt[1] + ".png"
        img_name = splt2
        cv.imwrite(img_name, img)
        count2 += 1
        if count2 % 2 == 0:
            count += 1

#copy_image(path)
rename_image2(path)
