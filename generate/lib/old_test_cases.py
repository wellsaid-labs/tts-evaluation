from dataclasses import dataclass, field
import string
import re
import random
import typing
from enum import Enum


def remove_xml(s: str):
    """Remove xml surrounding original word from string"""
    xml = re.compile(r"(<[^<>]+>)")
    xml_parts = xml.findall(s)
    for p in xml_parts:
        s = s.replace(p, "")
    return s

@dataclass
class TestCases:
    LONG_SCRIPTS = (
        (
            "“Yield Computations.”\nThe interest rate will always be stated as a percentage of the par "
            "value. The interest stated on the face of the bond is called the nominal yield. Sometimes "
            "it is referred to as the coupon rate. To compute the annual interest payments in dollars, "
            "multiply this nominal yield by the face amount of the bond (1000 dollars unless stated "
            "otherwise). A bond with a 5% coupon rate pays 50 dollars per year. One with an 8% nominal "
            "yield pays 80 dollars per year. One with a coupon of 13 point 5% pays 135 dollars per "
            "year. Because, on any particular bond, this interest payment is the same every year, it "
            "is referred to as a fixed charge.\n“Test Topic Alert.”\nWhen a question states that a "
            "bond pays interest at a rate of 6% semiannually, it does not mean two payments of 60 "
            "dollars per year. The interest rate is always stated on an annual basis (60 dollars per "
            "year), and it is paid twice per year, or 30 dollars every 6 months.\n“Current Yield "
            "(C.Y.)”\nInvestors always want to know the return on their investment. The most "
            "straightforward way to do that is to place the return on the investment as follows: "
            "Return divided by Investment.\nThe return will always be the annual interest in dollars "
            "(if referring to a stock, the dividend in dollars) divided by the current market price "
            "(the amount of investment required to own the security). This calculation is called "
            "current yield or current return. For example, if the annual dividend is 3 dollars and the "
            "stock’s price is 60 dollars the calculation is 3 divided by 5% equals 60 dollars.\n"
            "Although most bonds are issued with a face, or par, value of 1000 dollars bond prices do "
            "fluctuate in the market. As stated earlier, the interest a bond pays is called its coupon "
            "rate or nominal yield. Look at this example: The D B L tens of 39. DBL is the name of the "
            "issuer, tens means the nominal yield is 10%, and 39 means that the bonds mature in 20 39. "
            "The letter “s” is added because it is easier to say “the tens” than to say “the ten.” "
            "These bonds pay 100 dollars a year (50 dollars semiannually) for each 1000 dollars of "
            "face value. Regardless of what the market price of the bonds may be, the DBL Corporation "
            "has an obligation to pay annual interest of 10% of the 1000 dollars face it borrowed.\nIf "
            "an investor were to buy these bonds for more than 1000 dollars or less than 1,000 dollars "
            "the return on the investment would not be 10%. For example, if these bonds had a current "
            "market value of 800 dollars their current yield would be 12 point 5% (100 dollars divided "
            "by 800 dollars). Similarly, someone paying twelve hundred dollars for the bonds will "
            "receive a current yield of 8 point 3 3 percent (100 dollars divided by twelve hundred "
            "dollars). Please note that the 100 dollars interest received is the same in all cases "
            "regardless of the current market price.\nBond prices and yields move in opposite "
            "directions: as interest rates rise, bond prices fall, and vice versa. When a bond trades "
            "at a discount, its current yield increases; when it trades at a premium, its current "
            "yield decreases.\n“Discount and Premium.”\nWhen a bond is selling at a price above par "
            "(or face), it is selling at a premium; when it is selling below par, it is selling at a "
            "discount. Two critical statements to remember are:\nIf you pay more, you get less.\nIf "
            "you pay less, you get more.\nLooking at the examples above, an investor buying a bond at "
            "a premium will always receive a rate of return less than the coupon (or nominal) yield "
            "stated on the face of the bond (8 point 3 3 percent is less than 10%). Conversely, any "
            "time an investor purchases a bond at a discount, the return will be more than the rate "
            "stated on the face of the bond (12 point 5% is greater than 10%).\nIn addition to being "
            "the dollar amount on which the annual interest is based, par value is also the dollar "
            "amount that will be returned to the investor at maturity. Therefore, an investor "
            "purchasing a bond at a discount knows that holding the bond until maturity date will "
            "result in a return of the par value, an amount which will be more than what was paid for "
            "the bond.\nthat the par value received will be less than what was paid for the bond. To "
            "accurately reflect this gain or loss that an investor will have upon maturity, there is "
            "another yield to consider the yield to maturity, or true yield.\n\n"
        ),
        (
            "“Lesson 9 point 4: Financial And Recordkeeping Requirements For Investment Advisers.”\n"
            "“Learning Objective 9 eff.” Recognize the financial requirements to register as an "
            "investment adviser.\nUnder the Investment Advisers Act of 19 40, there are no specific "
            "financial requirements, such as a minimum net worth. However, as we will see, there are "
            "financial disclosures that must be made to clients under certain conditions.\n“Investment "
            "Adviser Financial Requirements.”\nUnder the Uniform Securities Act, the Administrator "
            "may, by rule or order, establish minimum financial requirements for an investment adviser "
            "registered in the state. We will begin with some requirements that apply to both federal "
            "covered and state-registered investment advisers.\n“Substantial Prepayment of Fees.”\n"
            "Both state and federal law offer extra protection to those clients of investment advisers "
            "who have made substantial advance payment of fees for services to be rendered in the "
            "future.\nThe term used is substantial prepayment of fees. In the case of a federal "
            "covered adviser, it is considered substantial if the IA collects prepayments of l twelve "
            "hundred dollars per client, six months or more in advance. Under the USA, it is more than "
            "500 dollars and again, six months or more in advance.\n“Test Topic Alert.”\nUnder the "
            "USA, when an investment adviser accepts prepayments of fees of more than 500 dollars for "
            "a contract period of six months or more, it is known as a substantial prepayment. "
            "However, under the Investment Advisers Act of 19 40, it does not become a substantial "
            "prepayment until it exceeds twelve hundred dollars.\n“Practice Question.”\nWhich of the "
            "following would nas-uh consider to be a substantial prepayment of fees?\n“A.” 500 dollars "
            "covering the next six months\n“B.” 800 dollars covering the entire contract year\n“C.” "
            "800 dollars covering the next calendar quarter\n“D.” 5,000 dollars covering the next month"
            "\n“The answer is bee.” nas-uh (state law) defines a substantial prepayment of fees to be "
            "more than 500 dollars six or more months in advance. While 800 dollars and 5,000 dollars "
            "are certainly more than 500 dollars they cover a shorter period than six months.\n"
            "“Disclosure of Financial Impairment.”\nAny investment adviser that has discretionary "
            "authority or custody of client funds or securities, or requires or solicits substantial "
            "prepayment of fees, must disclose any financial condition that is reasonably likely to "
            "impair its ability to meet contractual commitments to its clients. As an example, the "
            "S.E.C. has indicated that disclosure may be required of any arbitration award "
            "“sufficiently large that payment of it would create such a financial condition.”\nHere is "
            "the way it is stated in Form A.D.V. Part 2:\nIf you have discretionary authority or "
            "custody of client funds or securities, or you require or solicit prepayment of more than "
            "twelve hundred dollars in fees per client, six months or more in advance, you must "
            "disclose any financial condition that is reasonably likely to impair your ability to meet "
            "contractual commitments to clients.\n“Test Topic Alert.”\nThe Administrator may require "
            "an adviser who has custody of client funds or securities or has discretion over a "
            "client’s account to post a surety bond or maintain a minimum net worth. The requirement "
            "is higher for custody than for discretion. Typically, the net worth required of "
            "investment advisers with discretionary authority is 10,000 dollars and that for those "
            "taking custody, whether or not they are exercising discretion, is 35,000 dollars. If the "
            "adviser is using a surety bond instead, the requirement in either case is 35,000 dollars."
            "\nAn adviser who does not exercise discretion and does not maintain custody but does "
            "accept prepayment of fees of more than 500 dollars six or more months in advance, must "
            "maintain a positive net worth at all times.\n“Definition:” surety bond. A surety bond is "
            "usually issued by an insurance company (the surety) who guarantees payment of a specified "
            "sum to an injured party (either the client or the Administrator) when the securities "
            "professional causes damages by her actions or fails to perform.\n\n"
        ),
        (
            "Let’s learn how to process a wire transfer in TPSS.\n\nA banker must always identify the "
            "customer before authenticating in TPSS.\n\nAfter verifying the customer’s identity, let’s "
            "search for their account. There are multiple ways to search for a customer.\n\nIn this "
            "example, we’ve chosen to search for the customer’s account by searching with the Account "
            "Number.\n\nSearching by account number includes selecting the type of account.\n\nIn this "
            "example we are searching for a personal account, so the Personal Type is selected.\n\n"
            "Enter the account number.\n\nYou will authenticate the customer to ensure proper "
            "transaction authority. We can begin processing the customer’s wire transfer request from "
            "this account.\n\nSelect Outgoing Wire Transfer because the customer requests a wire "
            "transfer from this account.\n\nThis wire transfer will be sent to another bank (or "
            "financial institution) in the United States.\n\nConfirm with the customer which account "
            "to debit for the wire transfer.\n\nSome customer information will automatically populate "
            "once the account is selected.\n\nNow, enter the rowting number (known as ABA) of the bank "
            "where the funds will be sent and Wire Amount.\n\nThen select Calculate.\n\nNow, using the "
            "provided information, complete the remaining requestor and beneficiary entries.\n\nAs the "
            "Preparer, you need to enter your telephone number, and attest that the wire transfer has "
            "been reviewed and the customer has been identified.\n\nSelect prahgress.\n\nPrint two "
            "copies of the wire transfer request form.\n\nSelect a printer.\n\nProvide the printed "
            "Wire Transfer Request to the customer. The request should be signed by the customer after "
            "they review it.  The customer can leave the branch at this point.  They do not have to "
            "remain in branch for the approval process.\n\nSelect prahgress.\n\nThe wire transfer "
            "needs to be approved.\n\nAn authorized approver will approve the wire transfer by "
            "selecting the Work Items tab and selecting the Work Item ID. It is important to approve "
            "by wire cutoff time.\n\nPlease note there are thresholds for approving wires.  Over "
            "75,000 are sent to the Wire Support Unit for approval.\n\nThe Approver needs to enter "
            "their telephone number and attest that the wire transfer is reviewed and approved.\n\n"
            "Then, they’ll select prahgress.\n\nThe customer’s request has been completed. Select "
            "“End” to end this\nsession.\n\nIt is important to end each customer session when you are "
            "done\nto ensure information does not carry over to your next customer. "
        ),
        (
            "You and the warrior gehk-oh wake up after finishing the scroll, both at a loss for words "
            "after seeing all the violence and chaos that transpired. There is no longer any doubt, "
            "the guardians are the “lost” fifth faction. You waste no time reaching for the next "
            "scroll and realize its the last one, you’re overcome with a mixture of anxiety and "
            "excitement. Just as you start to read the scroll, you faintly hear a voice, but it fades "
            "out as you are once again immersed into the story.\n\nYou are in a large, makeshift "
            "facility and see a massive pile of shattered crystals upon a platform surrounded with "
            "unfamiliar technology. It immediately dawns on you what you’re seeing, the guardians "
            "enter the room and you watch as they activate some of the machines with the power from "
            "the crystals. The machines proh-ject an image of what is happening on Araka! They are "
            "starting to watch the revolt against the Emperor that you’ve already seen unfold. The "
            "old, wise gehk-oh you’ve seen in previous scrolls starts to speak, “We must try to stop "
            "this before things get out of hand. The downfall of the Emperor is overdue, but even if "
            "he is overthrown, I fear that afterwards each faction will be at each other’s throats "
            "before they are able to build a sufficient power grid.” Another replies, “Maybe we should "
            "have hidden more crystal shards there rather than taking nearly all of them so they would "
            "still be apart of our grid. Just look at all the destruction that the loss of power has "
            "caused…” The old, wise gehk-oh responds, “Its hard to say, but we couldn’t risk the "
            "Emperor finding shards, even the one piece we left was dangerous considering he could "
            "have harnessed its energy into some sort of weapon if he had found it. The Emperor was "
            "impossible to predict and becoming increasingly brazen with his antics, and furthermore, "
            "our own power grid still isn’t even running at its full capacity after the move. It is "
            "prudent that we do everything in our power to preserve the species. I do have a potential "
            "solution though and I’ve been working on prototypes for awhile now as a last resort. As "
            "he says this he lays out the schematics of his new invention and the rest of the "
            "guardians start to study his designs. You see the initial concepts for the concordians! "
            "He continues, “With these droids we could transmit a signal to prevent any more violence "
            "without becoming directly involved in the conflict.”"
        ),
        (
            "Church and community leaders in West Virginia, did you know that our state has one of the "
            "highest rates of children in foster care per capita in the nation? It's time to take "
            "action and follow the principles of James 1:27 to care for these vulnerable children. "
            'Join the "All In Foster Care Summit" on May 3rd, 2023, at River Ridge Church Tays Valley '
            "in Hurracun, West Virginia. This unique event unites church leaders, community members, "
            "and national and state child welfare experts to positively advance foster care in our "
            "state. Hear from prominent speakers and panelists about the role of faith-based and "
            "community efforts in foster care and how churches and communities can work together to "
            "transform foster care in their communities. Register in advance at see em vee doubleyou "
            "vee dot org and make a positive difference in the lives of vulnerable children and "
            "families. Come together and join the movement to improve foster care in West Virginia. "
            "Learn more at see em vee doubleyou vee dot org."
        ),
        (
            "Mana's March Candle is being resisted by Orange M A.\n\nIf the candle doesn't break "
            'through the 0.7 line strongly, we can expect a drop to 0.55 with a red ""MA""\n\n'
            "\n\n\n\nCandles are being resisted by 2 MA lines.\n\nIt seems difficult to break through "
            "the resistance line right now.\n\nIf it falls without breaking the 0.7 line, we can "
            "expect support at 0.55 first.\n\n\n\nCandles are stuck between MA lines.\n\nIt's a bad "
            "sign.\n\nCandles need to break through the 0.7 and 0.72 lines strongly to get to the bull "
            "market.\n\nOtherwise, it could fall to the 0.55 line.\n\nStokastic middle wave is also "
            "being bent.\n\nEven if the price falls in the short term, if the middle wave rises again, "
            "we can expect an additional increase along with the big wave.\n\n\n\nAs expected, candles "
            "are stuck between MA lines.\n\nAnd still being resisted by the Blue M A.\n\nIf candle "
            "leave the 0.62 line, you'd better prepare for a drop of 0.59 to 0.55.\n\nStokastic Big "
            "Wave also lacks strength.\n\nSo the candle must not deviate from the 0.62 line to rise."
        ),
        (
            "Picture a world where the truth is concealed, and history remains a mystery. What if "
            "there was a forbidden book, holding the key to explosive revelations of our ancestral "
            "past? The banned Book of Enoch unveils the shocking Truths that have been buried for "
            "centuries. Are you daring enough to venture into the unknown and unravel the secrets of "
            "our history? If so, Embark on a journey with us into the forbidden realm of the banned "
            "Book of Enoch, a text that has been hidden in secrecy for ages. This ancient scripture "
            "has been discredited, banned, and lost to the world. But now with the discovery of the "
            "Dead Sea Scrolls, its secrets can finally be uncovered. Welcome to our channel.  In this "
            "video: we will uncover the shocking secrets that have been kept hidden for years. Come "
            "with us as we embark on an adventure to uncover the mysteries that have been concealed "
            "for centuries. The Book of Enoch harbors numerous taboos that have led to its banishment "
            "and it being discredited by the mainstream."
        ),
        (
            "Would your Patients like to receive monthly Kovid-19 at-home test kits at No Cost, "
            "delivered straight to their door and increase your businesses bottom line ?  Delivered "
            "Labs is now enrolling Medical Professionals into an exciting program, which allows you to "
            "offer up to 8 Kovid-19 at-home test kits at No Cost to your Patients and shipped monthly, "
            "directly to their door, 100% covered by the Patients health insurance.\nOur turnkey "
            "program includes everything your business will need to successfully get started with "
            "ease, including automated order fulfillment software that can handle small to very large "
            "monthly Patient orders without disrupting your current business.\nHere at Delivered Labs, "
            "we’re simplifying your Patients needs by acquiring quality healthcare testing, all from "
            "the comfort of their homes.\nDon't miss out on our very lucrative and hands off Lab "
            "Testing program, to improve your patients' care. Sign up today and make testing for "
            "Kovid-19 easy and accessible for your patients!\n"
        ),
        (
            "As Momiji starts putting the boxed books inside the closet, & Mahiro offers to help but "
            "trips on her. This leads to the two of them falling into an awkward position. Mihari "
            "bursts in through the door to see what the noise was, and she sees Momiji’s hands placed "
            "on Mahiro’s flat chest. Everyone ends up laughing it off, and Kha-ehday comes to believe "
            "that Mahiro and Kha-ehday have become best friends now. All good things must come to an "
            "end, and it became time for Kha-ehday and Momiji to go back home. Kha-ehday thanks Mihari "
            "for helping her practice for her incoming test, and Momiji thanked Mahiro for spending "
            "time with her. As the two girls leave, Mihari commends Mahiro for being able to make a "
            "new friend, despite Mahiro’s disagreement that he technically is an adult in a girl’s "
            "body, and it’d be weird of him to be friends with a middle-school girl. Kha-ehday asks "
            "Momiji what she did with Mahiro, and Momiji tells her that it’s a secret, but she has "
            "grown to like Mahiro as a new friend."
        ),
        (
            "How do you master ignore someone?\n\nI know that. Mastering the art of ignoring someone "
            "can be difficult. but you know what. i think it is possible.\n\nThe first step you need "
            "to do is. start recognize when someone is “pushing your buttons”. and causing you to "
            "become angry or upset. Once you can recognize this. it will be easier to take a step "
            "back. Let’s take some deep breaths and remind yourself that. it is not worth your energy "
            "to engage with this person.\n\nAnother tip i think will help you  is. you should learn "
            "how to practice self-care. and find activities that will help you de-stress.  such as "
            "taking a walk. listening to music. or meditating.\n\nThis will help you to keep your "
            "emotions in check. so that you can remain calm and ignore the person.\n\nGuys we don’t "
            "have to let someone’s words or actions get to you. if you don’t want them to.\n\nJust "
            "remove negative person out of your life and you will good to go.\n\nI hope this video "
            "will help you!\n\nSee you next time! Bye Bye. I Love You!"
        ),
        (
            "The LiuGong S935 sugarcane harvester is a cutting-edge machine designed for use in the "
            "agricultural industry. With its robust construction and advanced features, this harvester "
            "can quickly and efficiently harvest large quantities of sugarcane.\nEquipped with a "
            "powerful engine, the LiuGong S935 is capable of reaching a maximum speed of 20 km/h, "
            "ensuring rapid movement across fields. The harvester's hydraulic system drives the "
            "cutting blades and feeder roller, allowing for precision cutting and harvesting of the "
            "sugarcane. The machine's unloading system can also easily unload the harvested sugarcane, "
            "making the process more efficient.\n\nThe cab of the LiuGong S935 is ergonomically "
            "designed for maximum comfort and control for the operator. With its large, easy-to-read "
            "control panel, the operator can easily adjust the speed, height, and direction of the "
            "harvester to ensure optimal performance. It is also equipped with an air conditioning, "
            "providing a comfortable environment to the driver.\n\n"
        ),
        (
            "This dish combines the traditional ingredients of carbonara - bacon, egg, and cheese - "
            "with a delicate pasta parcel, that is filled with a light carbonara sauce. It's a magical "
            "experience as you bite into the soft pasta and feel the flavors of the carbonara sauce "
            "explode in your mouth. This dish is a true testament to Beck's creativity and his ability "
            "to elevate the traditional flavors of Italian cuisine.\n\nLa Pergola is not only about "
            "its food but also its elegant and luxurious decor. With a combination of classic and "
            "contemporary interior design, the restaurant has a timeless elegance that is truly "
            "captivating. The ambiance is perfect for a romantic dinner with its subtle lighting and "
            "soft background music. The impeccable service is yet another highlight, making sure that "
            "your dining experience is a memory you will cherish for years to come.\nIn conclusion, if "
            "you are looking for a unique culinary experience with a breathtaking view of Rome, then "
            "La Pergola is the perfect destination."
        ),
        (
            "Are you unknowingly damaging your skin, with these common skincare mistakes? In this "
            "video, we'll share some of the most common things you should never put on your face.\n"
            "Hello everyone, and welcome back to Know-How. We all want to have healthy and glowing "
            "skin, but sometimes we make mistakes that can damage our skin. So, in this video, we’re "
            "going to share with you some of the products and ingredients, that you should avoid "
            "putting on your face at all costs.\n\n#1. Lemon Juice.\nLemon juice has become a popular "
            "natural remedy for various skin concerns, such as aknee, hyper-pigmentation, and dark "
            "spots. This is because lemon juice contains vitamin C and citric acid, which have "
            "skin-brightening and exfoliating properties. However, lemon juice can be very acidic, "
            "and applying it directly to your skin can cause more harm than good.\nThe high acidity of "
            "lemon juice can disrupt your skin's natural P-aitch balance, which can lead to dry-ness, "
            "redness, and even hyper-pigmentation. Additionally,"
        ),
        (
            "It's true that Disney has written some of the most romantic love stories in history. "
            "Disney romances often feature star-crossed lovers who are destined to be together, but "
            "some couples are brought together by the most unlikely of circumstances. Whatever their "
            "place of origin, there is nothing more magical than the Disney couples who, despite all "
            "odds, demonstrate that true love exists and that happiness is just around the corner at "
            "the most unexpected times.Hey guys, what’s up? It’s your channel, Amuse Crush. Make sure "
            "you hit that subscribe button and the bell icon to never ever miss an update from us in "
            "the future! Today, in this video, we’ll be counting down some of the Disney couples who "
            "could still be together.Number 5, Belle & Beast.What a timeless tale it is—a story that "
            "dates back to the beginning of time. Keeping someone in captivity is never acceptable, "
            "and that is what we want to say first. Because of this, there are some rather troubling "
            'aspects in "Beauty and the Beast."'
        ),
        (
            "We turned our backs on someone to express displeasure, anger, or disappointment, even if "
            "it wasn't someone in our own group. A narcissist, on the other hand, may exhibit similar "
            "behavior but with entirely opposite goals in mind. The silent treatment is a manipulative "
            "technique used to devalue the target's feelings and self-worth. Those who struggle with "
            "low self-esteem shouldn't have to go through life alone. You have zero worth without "
            "them. They will randomly stop talking to you, and you will have no choice except to "
            "grovel and apologize for everything.\n\nFor the fifth point, they claimed they were "
            "clueless. It's like putting your faith in a bad egg to trust a narcissist. Narcissists "
            "constantly put themselves first, even when things are at their worst.\n\nThe trouble is "
            "that they won't admit they were wrong and are always ready with an excuse. They could act "
            "carefree and cheerful to mask their true evil aim. You'll hear people say they weren't "
            "awful people, just naive or innocent."
        ),
    )
    # Yes/No questions expecting an upward inflection at the end of the sentence
    QUESTIONS = (
        "What is the meaning of life, the universe, and everything?",
        "Are birds real?",
        "Would you be a dear and make us a cup of tea?",
        "Must you be so inconsiderate?",
        "Have you considered the implications of faster-than-light travel?",
        "Is it true that the moon is made of cheese?",
        "Does that make sense?",
        "Can I help you?",
        "Are you suggesting that coconuts migrate?",
        "Can I get a backstage pass to the concert?",
        "Given that each of the following bonds is of the same quality and has the same maturity as the others, does one have a longer duration?",
        "Registration of an investment adviser automatically confers registration on?",
        "Church and community leaders in West Virginia: did you know that our state has one of the highest rates of children in foster care per capita in the nation?",
        "Are you daring enough to venture into the unknown and unravel the secrets of our history?",
        "Would your Patients like to receive monthly Covid-19 at-home test kits at no cost, delivered straight to their door?",
        "Are you unknowingly damaging your skin with these common skincare mistakes?",
        "Yes of course, but shall I introduce myself first?",
        "Do you consider yourself an adventurous person?",
        "Have you ever hidden a snack so that nobody else would find it and eat it first?",
        "How do we propose to end the scourge of liens ignorance, you ask?",
    )
    INITIALISMS = (
        "E3, Electronic Entertainment Expo, is an annual video games conference.",
        "IDK what you mean.",
        "The WHO has declared COVID-19 a global pandemic.",
        "My shady uncle works for the NSA.",
        "NHL allstar Wayne Gretzky played for four teams over the course of his career.",
        "Open up, FBI!",
        "This chart shows the SPE software initialization process as it connects to the hardware and transmit is enabled.",
        "There is an MMORPG for almost every major intellectual property.",
        "The 18 BUN dialog box is displayed.",
        "While many Swayj-lock products are classified E.A.R. 99, some Swayj-lock products have ECCNs that require government "
        "licenses, permissions, or other certifications prior to export.",
        "While many Swayj-lock products are classified EAR-99, some Swayj-lock products have ECCNs that require government "
        "licenses, permissions, or other certifications prior to export.",
        "Classification occurs prior to beginning the manufacturing process for ETOP, NPD, and Custom Solutions.",
        "Research tells us that induction process accounts for needs of Aboriginal and CALD young people.",
        "It was a tale of two halves. One half, where Eagles quarterback, Jaylen Hurts, was well on his way to the Super Bowl MVP.",
        "GHG Protocol (or GHGP) was developed in partnership with the World Resources Institute (WRI) "
        "and the World Business Council for Sustainable Development (WBCSD).",
        "Now, this system has generated an ACH file, which includes five beneficiaries for which the total amount "
        "of five thousand three hundred ninety-six dollars is being paid.",
        "These sections are located where the smaller IFAs on the narrower left floodplain indicate that flow "
        "in the left floodplain contracts and expands over a shorter distance.",
        "It took six Ph.Ds to design a VCR a five-year-old could use.",
    )
    URLS = (
        "en.wikipedia.org",
        "linkedin.com",
        "stackoverflow.com",
        "zillow.com",
        "twitch.tv",
        "zoom.us",
        "amazon.de",
        "bbc.co.uk",
        "nih.gov",
        "dailymail.co.uk",
        "imdb.com",
        "nytimes.com",
        "ikea.com",
        "MSDSsource.com",
        "alliancepropertysystems.com",
        "ion2.nokia.com",
    )
    URLS_HTTPS = tuple([f"https://www.{i}" for i in URLS])
    URLS_IN_SENTENCES = tuple(
        [f"Visit our website at {i} for more information." for i in URLS]
    )
    URLS_HTTPS_IN_SENTENCES = tuple(
        [f"Visit our website at {i} for more information." for i in URLS_HTTPS]
    )
    EMAILS = (
        "tracy.madison@CSWPBC.com",
        "kay_sorensen@insurewithcompass.com",
    )
    EMAILS_IN_SENTENCES = tuple(
        [f"Please email {i} if you need further support." for i in EMAILS]
    )
    RESPELLINGS: typing.Dict = field(
        default_factory=lambda: {
            "acetaminophen": (
                "<respell value='uh-SEE-tuh-MIH-nuh-fuhn'>acetaminophen</respell>",
                "::uh-SEE-tuh-MIH-nuh-fuhn::",
            ),
            "adobe": (
                "<respell value='uh-DOH-bee'>adobe</respell>",
                "::uh-DOH-bee::",
            ),
            "antihistamines": (
                "<respell value='AN-tee-HIH-stuh-muhnz'>antihistamines</respell>",
                "::AN-tee-HIH-stuh-muhnz::",
            ),
            "asphalt": (
                "<respell value='AS-fawlt'>asphalt</respell>",
                "::AS-fawlt::",
            ),
            "asthma": (
                "<respell value='AZ-muh'>asthma</respell>",
                "::AZ-muh::",
            ),
            "asymptomatic": (
                "<respell value='AY-SIHMP-tuh-MA-tihk'>asymptomatic</respell>",
                "::AY-SIHMP-tuh-MA-tihk::",
            ),
            "bluish": (
                "<respell value='BLOO-ihsh'>bluish</respell>",
                "::BLOO-ihsh::",
            ),
            "crises": (
                "<respell value='KRY-seez'>crises</respell>",
                "::KRY-seez::",
            ),
            "denials": (
                "<respell value='dih-NY-uhlz'>denials</respell>",
                "::dih-NY-uhlz::",
            ),
            "deny": ("<respell value='dih-NY'>deny</respell>", "::dih-NY::"),
            "duo": ("<respell value='DOO-oh'>duo</respell>", "::DOO-oh::"),
            "enrolled": (
                "<respell value='ehn-ROHLD'>enrolled</respell>",
                "::ehn-ROHLD::",
            ),
            "flange": ("<respell value='FLANJ'>flange</respell>", "::FLANJ::"),
            "goldilocks": (
                "<respell value='GOHL-dee-lawks'>goldilocks</respell>",
                "::GOHL-dee-lawks::",
            ),
            "hypoglycemic": (
                "<respell value='HY-poh-gly-SEE-mihk'>hypoglycemic</respell>",
                "::HY-poh-gly-SEE-mihk::",
            ),
            "icon": ("<respell value='Y-kahn'>icon</respell>", "::Y-kahn::"),
            "incorporate": (
                "<respell value='ihn-KOR-pur-ayt'>incorporate</respell>",
                "::ihn-KOR-pur-ayt::",
            ),
            "iodine": (
                "<respell value='Y-uh-dyn'>iodine</respell>",
                "::Y-uh-dyn::",
            ),
            "karaoke": (
                "<respell value='KEH-ree-OH-kee'>karaoke</respell>",
                "::KEH-ree-OH-kee::",
            ),
            "meteor": (
                "<respell value='MEE-tee-ur'>meteor</respell>",
                "::MEE-tee-ur::",
            ),
            "niche": ("<respell value='NIHCH'>niche</respell>", "::NIHCH::"),
            "nomenclature": (
                "<respell value='NOH-muhn-klay-chur'>nomenclature</respell>",
                "::NOH-muhn-klay-chur::",
            ),
            "omit": ("<respell value='oh-MIHT'>omit</respell>", "::oh-MIHT::"),
            "oodles": (
                "<respell value='OO-duhlz'>oodles</respell>",
                "::OO-duhlz::",
            ),
            "osteoporosis": (
                "<respell value='AW-stee-OH-pur-OH-sihs'>osteoporosis</respell>",
                "::AW-stee-OH-pur-OH-sihs::",
            ),
            "pancreas": (
                "<respell value='PAN-kree-uhs'>pancreas</respell>",
                "::PAN-kree-uhs::",
            ),
            "quarantine": (
                "<respell value='KWOH-ruhn-teen'>quarantine</respell>",
                "::KWOH-ruhn-teen::",
            ),
            "realm": ("<respell value='REHLM'>realm</respell>", "::REHLM::"),
            "reconnaissance": (
                "<respell value='ree-KAH-nuh-suhns'>reconnaissance</respell>",
                "::ree-KAH-nuh-suhns::",
            ),
            "rendezvous": (
                "<respell value='RAHN-dih-voo'>rendezvous</respell>",
                "::RAHN-dih-voo::",
            ),
            "ricocheted": (
                "<respell value='RIH-kuh-shayd'>ricocheted</respell>",
                "::RIH-kuh-shayd::",
            ),
            "scarce": (
                "<respell value='SKERRS'>scarce</respell>",
                "::SKERRS::",
            ),
            "sew": ("<respell value='SOH'>sew</respell>", "::SOH::"),
            "signage": (
                "<respell value='SY-nihj'>signage</respell>",
                "::SY-nihj::",
            ),
            "sterile": (
                "<respell value='STEH-ruhl'>sterile</respell>",
                "::STEH-ruhl::",
            ),
            "teammate": (
                "<respell value='TEE-mayt'>teammate</respell>",
                "::TEE-mayt::",
            ),
            "trachea": (
                "<respell value='TRAY-kee-uh'>trachea</respell>",
                "::TRAY-kee-uh::",
            ),
            "treachery": (
                "<respell value='TREH-chur-ee'>treachery</respell>",
                "::TREH-chur-ee::",
            ),
            "triage": (
                "<respell value='TREE-ahj'>triage</respell>",
                "::TREE-ahj::",
            ),
            "verbatim": (
                "<respell value='vur-BAY-tuhm'>verbatim</respell>",
                "::vur-BAY-tuhm::",
            ),
            "feisty": (
                "<respell value='FY-stee'>feisty</respell>",
                " ::FY-stee::",
            ),
            "corps": ("<respell value='KOR'>corps</respell>", "::KOR::"),
            "ague": ("<respell value='AY-gyoo'>ague</respell>", "::AY-gyoo::"),
            "saga": ("<respell value='SAH-guh'>saga</respell>", "::SAH-guh::"),
            "heist": ("<respell value='HYST'>heist</respell>", "::HYST::"),
            "overzealous": (
                "<respell value='OH-vur-ZEH-luhs'>overzealous</respell>",
                "::OH-vur-ZEH-luhs::",
            ),
            "zeal": ("<respell value='ZEEL'>zeal</respell>", "::ZEEL::"),
            "juxtaposition": (
                "<respell value='JUHK-stuh-puh-ZIH-shuhn'>juxtaposition</respell>",
                "::JUHK-stuh-puh-ZIH-shuhn::",
            ),
            "overexaggerated": (
                "<respell value='OH-vur-ihg-ZA-juh-ray-duhd'>overexaggerated</respell>",
                "::OH-vur-ihg-ZA-juh-ray-duhd::",
            ),
            "onomatopoeia": (
                "<respell value='AH-nuh-MAH-duh-PEE-uh'>onomatopoeia</respell>",
                "::AH-nuh-MAH-duh-PEE-uh::",
            ),
            "California": (
                "<respell value='KA-luh-FOR-nyuh'>California</respell>",
                "::KA-luh-FOR-nyuh::",
            ),
            "Indian": (
                "<respell value='IHN-dee-uhn'>Indian</respell>",
                "::IHN-dee-uhn::",
            ),
            "Italian": (
                "<respell value='ih-TA-lyuhn'>Italian</respell>",
                "::ih-TA-lyuhn::",
            ),
            "application": (
                "<respell value='A-pluh-KAY-shuhn'>application</respell>",
                "::A-pluh-KAY-shuhn::",
            ),
            "authority": (
                "<respell value='uh-THOH-ruh-tee'>authority</respell>",
                "::uh-THOH-ruh-tee::",
            ),
            "becoming": (
                "<respell value='bih-KUH-mihng'>becoming</respell>",
                "::bih-KUH-mihng::",
            ),
            "board": ("<respell value='BORD'>board</respell>", "::BORD::"),
            "car": ("<respell value='KAR'>car</respell>", "::KAR::"),
            "civil": (
                "<respell value='SIH-vuhl'>civil</respell>",
                "::SIH-vuhl::",
            ),
            "cover": (
                "<respell value='KUH-vur'>cover</respell>",
                "::KUH-vur::",
            ),
            "critical": (
                "<respell value='KRIH-tih-kuhl'>critical</respell>",
                "::KRIH-tih-kuhl::",
            ),
            "decision": (
                "<respell value='dih-SIH-zhuhn'>decision</respell>",
                "::dih-SIH-zhuhn::",
            ),
            "determined": (
                "<respell value='dih-TUR-muhnd'>determined</respell>",
                "::dih-TUR-muhnd::",
            ),
            "develop": (
                "<respell value='dih-VEH-luhp'>develop</respell>",
                "::dih-VEH-luhp::",
            ),
            "device": (
                "<respell value='dih-VYS'>device</respell>",
                "::dih-VYS::",
            ),
            "distribution": (
                "<respell value='DIHS-truh-BYOO-shuhn'>distribution</respell>",
                "::DIHS-truh-BYOO-shuhn::",
            ),
            "efforts": (
                "<respell value='EH-furts'>efforts</respell>",
                "::EH-furts::",
            ),
            "element": (
                "<respell value='EH-luh-muhnt'>element</respell>",
                "::EH-luh-muhnt::",
            ),
            "famous": (
                "<respell value='FAY-muhs'>famous</respell>",
                "::FAY-muhs::",
            ),
            "finally": (
                "<respell value='FY-nuh-lee'>finally</respell>",
                "::FY-nuh-lee::",
            ),
            "fully": (
                "<respell value='FUU-lee'>fully</respell>",
                "::FUU-lee::",
            ),
            "gives": ("<respell value='GIHVZ'>gives</respell>", "::GIHVZ::"),
            "giving": (
                "<respell value='GIH-vihng'>giving</respell>",
                "::GIH-vihng::",
            ),
            "growing": (
                "<respell value='GROH-ihng'>growing</respell>",
                "::GROH-ihng::",
            ),
            "hard": ("<respell value='HARD'>hard</respell>", "::HARD::"),
            "ideas": (
                "<respell value='y-DEE-uhz'>ideas</respell>",
                "::y-DEE-uhz::",
            ),
            "instance": (
                "<respell value='IHN-stuhns'>instance</respell>",
                "::IHN-stuhns::",
            ),
            "instead": (
                "<respell value='ihn-STEHD'>instead</respell>",
                "::ihn-STEHD::",
            ),
            "leaving": (
                "<respell value='LEE-vihng'>leaving</respell>",
                "::LEE-vihng::",
            ),
            "moving": (
                "<respell value='MOO-vihng'>moving</respell>",
                "::MOO-vihng::",
            ),
            "musical": (
                "<respell value='MYOO-zih-kuhl'>musical</respell>",
                "::MYOO-zih-kuhl::",
            ),
            "notes": ("<respell value='NOHTS'>notes</respell>", "::NOHTS::"),
            "nuclear": (
                "<respell value='NOOK-lee-ur'>nuclear</respell>",
                "::NOOK-lee-ur::",
            ),
            "ordered": (
                "<respell value='OR-durd'>ordered</respell>",
                "::OR-durd::",
            ),
            "output": (
                "<respell value='OWT-puut'>output</respell>",
                "::OWT-puut::",
            ),
            "phase": ("<respell value='FAYZ'>phase</respell>", "::FAYZ::"),
            "plants": (
                "<respell value='PLANTS'>plants</respell>",
                "::PLANTS::",
            ),
            "previously": (
                "<respell value='PREE-vee-uhs-lee'>previously</respell>",
                "::PREE-vee-uhs-lee::",
            ),
            "purpose": (
                "<respell value='PUR-puhs'>purpose</respell>",
                "::PUR-puhs::",
            ),
            "quite": ("<respell value='KWYT'>quite</respell>", "::KWYT::"),
            "race": ("<respell value='RAYS'>race</respell>", "::RAYS::"),
            "relations": (
                "<respell value='ree-LAY-shuhnz'>relations</respell>",
                "::ree-LAY-shuhnz::",
            ),
            "responsible": (
                "<respell value='ree-SPAHN-suh-buhl'>responsible</respell>",
                "::ree-SPAHN-suh-buhl::",
            ),
            "risk": ("<respell value='RIHSK'>risk</respell>", "::RIHSK::"),
            "rock": ("<respell value='RAHK'>rock</respell>", "::RAHK::"),
            "sets": ("<respell value='SEHTS'>sets</respell>", "::SEHTS::"),
            "share": ("<respell value='SHERR'>share</respell>", "::SHERR::"),
            "shot": ("<respell value='SHAHT'>shot</respell>", "::SHAHT::"),
            "situation": (
                "<respell value='SIH-choo-AY-shuhn'>situation</respell>",
                "::SIH-choo-AY-shuhn::",
            ),
            "something": (
                "<respell value='SUHM-thihng'>something</respell>",
                "::SUHM-thihng::",
            ),
            "table": (
                "<respell value='TAY-buhl'>table</respell>",
                "::TAY-buhl::",
            ),
            "techniques": (
                "<respell value='tehk-NEEKS'>techniques</respell>",
                "::tehk-NEEKS::",
            ),
            "unique": (
                "<respell value='yoo-NEEK'>unique</respell>",
                "::yoo-NEEK::",
            ),
            "vote": ("<respell value='VOHT'>vote</respell>", "::VOHT::"),
            "weeks": ("<respell value='WEEKS'>weeks</respell>", "::WEEKS::"),
            "win": ("<respell value='WIHN'>win</respell>", "::WIHN::"),
            "workers": (
                "<respell value='WUR-kurz'>workers</respell>",
                "::WUR-kurz::",
            ),
            "world": ("<respell value='WURLD'>world</respell>", "::WURLD::"),
        }
    )
    SLURRING = (
        (
            "If you’ve ever cyberbullied someone, you may have thought it was funny at the time. "
            "When you can’t see someone’s face, it can be easier to be mean to them. But chances "
            "are, you don’t really want to cause anyone to experience anything on the list above. "
            "It’s important to use empathy for others, even when we’re online and can’t look them "
            "in the eyes. Empathy is when we put ourselves in another person’s shoes to identify "
            "with how they are feeling. The next time you’re thinking about being disrespectful "
            "online just to get a laugh, put yourself in the other person’s shoes. Ask yourself, "
            "would you like someone to treat you that way?",
        ),
        (
            "After a couple of minutes, Gaius groaned once more and opened his eyes. “What… what "
            "happened?” he asked, as he immediately tried to sit up. Leon, pushing him back down, said,"
            " “We got the shadow cat, but you seemed to have passed out. How are you feeling?” “Uggh, "
            "like I fought a bull and lost,” Gaius replied, as he shifted around enough to pull his "
            "shirt up and see that the wound Leon had left on his abdomen had closed over with "
            "freshly-healed skin. “At least that’s taken care of. Thank you, Leon.”",
        ),
        (
            "Hi, I'm Idris. I'm going to guide you through this session today. I'm so excited to have "
            "you here to learn about underwriting recreational vehicles for our customers! I'd like "
            "to get to know you before we start; can you enter your name, please?",
        ),
        (
            "Let’s look at a couple of questions to reinforce what we have learned. You can pause the "
            "webcast to give yourself time to answer the question. Then, press 'play' to see if you "
            "answered correctly and to hear the rationale. Keep in mind there may be more than one "
            "answer. Number 1. In an ICD-10 PCS cabbage procedure code, what does the Body Part value "
            "identify? The name of the body part being bypassed. The number of coronary arteries being "
            "treated. The number of coronary arteries being bypassed to. The vessel bypassed from.",
            "We're sorry for the delay. Our average wait time is well under a minute, so we will be "
            "with you shortly. Did you know that we perform custom alterations for many of our "
            "customers? That’s right. Do you need a shorter than standard inseam, but can't find that "
            "service elsewhere? All of our pants are hemmed on-site free of charge, while adding a "
            "zipper to the leg of any of our pants can make accessing a knee brace or prosthesis so "
            "much easier. Most alterations add only one day to your order turnaround, so ask your "
            "order taker for more information. We will be with you shortly.",
        ),
        (
            "It's time to take what you've learned about DocuSign and put it into practice. In this "
            "course, you will learn how to access the demo environment, and have the opportunity to "
            "send practice documents.. both with a template and without a template. It will take "
            "approximately 10 minutes to complete and the course does contain audio, so please make "
            "sure your sound is turned on. To begin, click the start button.",
        ),
    )
    V10_REGRESSIONS = (
        # NOTE: Respellings are formatted like they were inputted in v10
        # - Difficult acronyms
        "It took six Ph.Ds to design a VCR a five-year-old could use.",
        # - "Cape Cod" was repeated
        "It is ironic that today's least "
        "<respell value='PAH-pyuh-lay-tuhd'>|\\PAH\\pyuh\\lay\\tuhd\\|</respell> town on Cape Cod",
        # - Short sentences were cut off
        "Taking sides early - I feel like... I feel like that's a recipe for disaster. It is.",
        "manager. Egan",
        "then walked away without taking any questions. Wow,",
        "Thanks! For..",
        "using your ears. Why?",
        "Yes. Are you ready to play? Yeah.",
        # - This question generated a long silence, after "morning"
        "Can you tell me more about what happened that morning?",
        # - This word was pronounced incorrectly
        "anemone",
        # - This word caused the model to overflow
        "<respell value='po-lahn-co'>|\\po\\lahn\\co|</respell>",
        "<respell value='fran-SIH-skoh'>|\\fran\\SIH\\skoh|</respell>",
        # ("Wii")
        "The Wii gaming console.",
        # Paige, Jarvis ("STOP")
        "This needs to STOP now.",
        # Ava ("queue")
        "We were forced to stand in a queue.",
        # Ava ("demo")
        "She will be showing a demo of the company's new alarm system.",
        # Donna, Wade ("Meet" dropping t sound)
        "We arranged to meet for lunch.",
        # Tilda, Lee, Vanessa, Jeremy, Nicole ("ChatGPT")
        "ChatGPT is a powerful chatbot developed by OpenAI that uses machine learning to generate \
          human-like responses to user input.",
        "There are several benefits to using text-to-speech with ChatGPT.",
        "Another benefit to using text-to-speech with ChatGPT is that it can make the chatbots' \
          responses sound more natural and human-like.",
        "One way to enhance the user experience with ChatGPT is by using a text to speech system to \
          convert the chatbot’s responses from text to speech, allowing users to hear the chatbot’s \
            responses instead of reading them.",
        # Phone numbers
        "My phone number is 723-5670",
        # Parenthesis
        "However when part of my script talked about a 401(k) plan and the voiceover pronounced it \
          four hundred and wonk",
        "I work for a financial industry and am making lots of videos talking about 401(k)s, 457(b)s, \
          and so on.",
    )
    # NOTE Slack Reference: https://wellsaidlabs.slack.com/archives/C0149LB6LKX/p1671134275497839
    # Wade ("EVV" pronounced with inconsistent speed)
    V10_ACCENTURE = (
        "Step 2: The EVV vendor reviews the EVV Provider Onboarding Form and confirms all required "
        "fields are complete and accurate.",
        "Within one business day of receipt, the EVV vendor will send an email to the signature "
        "authority and Program Provider/FMSA EVV System Administrator listed on the form to "
        "acknowledge receipt of the EVV Provider Onboarding Form.",
        "The EVV vendor will advise that the submitted form is under review and the contact "
        "information for the Program Provider/FMSA EVV System Administrator on the form will be "
        "used to contact  the program provider or FMSA to begin the EVV Provider Onboarding "
        "Process."
    )
    V10_SYNTHESIA = (
        # Slack Reference: https://wellsaidlabs.slack.com/archives/C0149LB6LKX/p1673021131011249
        # "AI", "Xelia", "studio" requesting v10 downgrade to v9
        "I’m an AI avatar",
        "Mindblowing AI tools you’ve never heard of",
        "We are happy to support tools supporting Xelia grows",
        "Here’s a quick overview of our studio platform",
    )
    VARIOUS_INITIALISMS = (
        # NOTE This section will be reused for testing v11 & all examples here are copied in intialisms section above
        "Each line will have GA Type as Payment, Paid Amount along with PAC, and the GA Code.",
        "Properly use and maintain air-line breathing systems and establish a uniform procedure "
        "for all employees, both LACC and LCLA contractors, to follow when working jobs that "
        "require the use of fresh air.",
        "QCBS is a method of selecting transaction advisors based on both the quality of their "
        "technical proposals and the costs shown in their financial proposals.",
        "HSPs account for fifteen to twenty percent of the population.",
        "We used to have difficulty with AOA and AMA, but now we are A-okay.",
        "As far as AIs go, ours is pretty great!",
    )
    QUESTIONS_WITH_UPWARD_INFLECTION = (
        # NOTE: All these questions should have an upward inflection at the end.
        "Have you ever hidden a snack so that nobody else would find it and eat it first?",
        "Can fish see air like we see water?",
        "Are you a messy person?",
        "Did you have cats growing up?",
        "Do you consider yourself an adventurous person?",
        "Do you like any weird food combos?",
        "Do you respond to texts fast?",
        "Have you ever been stalked by an animal that later became your pet?",
        "If you have made it this far, do you relate to any of these signs? Are you a highly "
        "sensitive person?",
        "Have you started, but not found success, with a platform requiring monthly payments?",
        "When deciding between organic and non-organic coffees, is the price premium worth it?",
        "Can you make yourself disappear?",
        "Do mice really eat cheese?",
        "Do you believe in any conspiracy theories?",
        "Have elves always lived at the North Pole?",
        "Have you ever been on the radio?",
        "Have you ever done something embarrassing in front of the office CCTV cameras?",
        "In your opinion, are giant spiders worse than giant chickens?",
        "Would you like to be part of the UK Royal Family?",
        "Have you ever tried any DIY projects?",
        "Can people from NASA catch the flu?",
        "Do you watch ESPN at night?",
        "Will AI replace humans?",
        "Can our AI say AI?",
    )
    QUESTIONS_WITH_VARIED_INFLECTION = (
        # NOTE: These questions each have a different expected inflection.
        "If you can instantly become an expert in something, what would it be?",
        "What led to the two of you having a disagreement?",
        "Why do some words sound funny to us?",
        "What are your plans for dealing with it?",
        "There may be times where you have to RDP to a node and manually collect logs for some "
        "reason. So, another question you may have is, exactly where on disk are all these logs?",
        "How common are HSPs?",
        "If you could rid the world of one thing, what would it be?",
        "What childish things do you still do as an adult?",
        "If you were to perform in the circus, what would you do?",
        "Who is eligible for reimbursement?",
    )
    # These statements have a mix of heteronyms, initialisms, hard words (locations,
    # medical terms, technical terms), etc for testing pronunciation.
    HARD_SCRIPTS = (
        "For more updates on covid nineteen, please contact us via the URL at the bottom of the "
        "screen, or visit our office in Seattle at the address shown here.",
        "I've listed INTJ on my resume because it's important for me that you understand how I "
        "conduct myself in stressful situations.",
        "The website is live and you can access your records via the various APIs/URLs or use "
        "the Studio as an alternate avenue.",
        "The nurses will resume the triage conduct around the oropharyngeal and test for "
        "tachydysrhythmia to ensure the patient lives another day.",
        "Access your clusters using the Kubernetes API. You can alternate between the CLI and the "
        "web interface.",
        "Live from Seattle, it's AIQTV, with the governor's special address on the coronavirus. Don't "
        "forget to record this broadcast for viewing later.",
        "Let's add a row on our assay tracking sheet so we can build out the proper egress "
        "measurements.",
        "Hello! Can you put this contractor into a supervisory role?",
    )
    # Test cases with a variety of lengths, respellings, and punctuation marks.
    HARD_SCRIPTS_2 = (
        "WellSaid Labs.",
        "Livingroom",
        "ACLA.",
        # NOTE: `ACLA` sometimes gets cut-off, this is a test to see how a period affects it.
        "NASA",
        "Topic two:     Is an NRA right for my rate?.",
        "Internet Assigned Numbers Authority (IANA)",
    )
    GREEK_SYMBOLS = (
        "Α α Β β Γ γ Δ δ Ε ε Ζ ζ Η η Θ θ Ι ι  Κ κ Λ λ Μ μ Ν ν Ξ ξ Ο ο Π π Ρ ρ Σ σ ς Τ τ Υ υ Φ φ Χ "
        "χ Ψ ψ Ω ω",
    )
    # Long strings of repetitive letters can highlight model instability.
    ALPHABET = (
        string.ascii_uppercase * 15,
        " ".join(list(string.ascii_uppercase) * 15),
        ". ".join(list(string.ascii_uppercase) * 15),
    )
    # short words and large annotation values can cause speaker switching
    SPEAKER_SWITCHING = (
        "Yeah",
        "Lol",
        '<tempo value="0.25">Please do not do that.</tempo>',
        '<tempo value="5">Please do not do that.</tempo>',
    )
    ABBREVIATIONS_WITH_VOWELS = (
        "ABBA (musical group) - Agnetha, Björn, Benny, Anni-Frid (first names of the band’s members)",
        "AFK - Away From Keyboard",
        "AFL – American Football League",
        "AGI - Artificial General Intelligence",
        "AWOL - Absent WithOut Leave",
        "CSI - Crime Scene Investigation",
        "DIY - Do It Yourself",
        "EOW - End of Week",
        "FAQ - Frequently Asked Questions",
        "PAWS – Progressive Animal Welfare Society",
        "POTUS - President of the United States",
        "POW - Prisoner Of War",
        "SCOTUS - Supreme Court of the United States",
        "TBA - To Be Announced",
        "TTYL - Talk To You Later",
        "WTH - What The Heck (or Hell)",
        "WWE – World Wrestling Entertainment",
        "YAHOO (search engine) - Yet Another Hierarchical Officious Oracle",
        "YOLO - You Only Live Once",
        "ZIP code - Zone Improvement Plan code",
    )
    SLOW_SCRIPTS = (
        (
            '<tempo value="0.3">Mirabeau B. Lamar was born in Georgia in 1798'
            "</tempo>. He was the son of a wealthy plantation owner. When he "
            'was grown, <tempo value="0.3">Lamar entered Georgia politics</tempo> and served on the '
            'state senate. After <tempo value="0.3">visiting a friend in Texas in 1834</tempo>, he '
            "decided he wanted to live there."
        ),
        (
            '<tempo value="0.4">98 percent of businesses</tempo> estimate that a <tempo value="0.4"> '
            'single hour</tempo> of downtime costs them more than <tempo value="0.4">100,000 dollars'
            "</tempo>."
        ),
        (
            'Ray decided to build a <tempo value="0.7">similar device for his own home</tempo>, and '
            'when he tested it, he was not only did his power spikes go away, but <tempo value="0.6">'
            "his power bill was almost cut in HALF!</tempo>"
        ),
        (
            'Whether it\'s puppy classes <tempo value="0.4"> or playdates with '
            "other dogs</tempo>, exposing your Husky to new experiences can help them to become a "
            '<tempo value="0.7">well-adjusted and friendly companion</tempo>.'
        ),
        (
            'How comfortable will you be in applying what you have learned about <tempo value="0.5">'
            "hydraulic design considerations</tempo> to your next bridge project? Drag the slider from "
            '0 to 10, with <tempo value="0.5"> 10 being the most comfortable</tempo>.'
        ),
        (
            'Are you <tempo value="0.7">drowning in tax burden and don\'t know '
            "what to do?</tempo> Well, bankruptcy might be an option for you, "
            'but it\'s important to understand that <tempo value="0.7">not all taxes can be eliminated '
            "through bankruptcy</tempo>."
        ),
        (
            "And one of the most impactful ways we help you connect with fans is through our "
            '<tempo value="0.6">powerful community of subscribers</tempo>. In fact, we recently '
            '<tempo value="0.7">surpassed 200 million</tempo>. That’s about <tempo value="0.6">3 '
            "percent of the world’s population</tempo> paying monthly so they can have uninterrupted "
            'access to your art. <tempo value="0.3">This is huge</tempo>.'
        ),
        (
            "As you review the details of the scenario, consider the following:\n\nKnowledge of "
            '<tempo value="0.8">current plans, capabilities, and insights</tempo>.\n\nMultiple <tempo '
            'value="0.8">options, actions, and solutions</tempo> for discussion. \n\nNow, let’s see '
            "how this all unfolds and how you and your organization would respond."
        ),
        (
            'Hey this is Major Simpson, the duty officer here at the <tempo value="0.5">six eighteenth '
            "A-O-C, for mission-number alpha-three-seven-five-zero</tempo>. Tell the crew that they’re "
            'done for the day, and we’re going to set them up for a <tempo value="0.5">'
            "two-two-three-zero Zulu</tempo> departure tomorrow.  Have the crew call me A-S-A-P please."
        ),
        (
            "When you are assigned an assessment assessment, take a moment to look at the content "
            'covered in the assessment.  Ask yourself: \n•        Are there <tempo value="0.5">any '
            "resources I will need</tempo> to complete this assessment?\n•        Are there "
            '<tempo value="0.5">any modules I can review</tempo> about this material?\n•        Would '
            'it be beneficial to <tempo value="0.5">read through work instructions or reference guides'
            "</tempo> on this content?"
        ),
        (
            'Organisations should ensure that the tools and equipment are <tempo value="0.6">'
            "thoroughly checked for cuts, cracks, and damages on cords and wires</tempo>. Replacing or "
            'repairing the damaged equipment should be done <tempo value="0.3">immediately</tempo>. '
            'If there is exposure to damaged tools, it is <tempo value="0.4">extremely dangerous'
            "</tempo>."
        ),
        (
            '<tempo value="0.8">Stand tall</tempo> and open your arms <tempo value="0.8">straight to '
            "the side</tempo>. Reach through your fingers and straighten your elbows "
            '<tempo value="0.8">as best you can</tempo>. Slowly start making circles at about the size '
            'of a coconut. Keep your <tempo value="0.8">arms back</tempo> in line with the shoulders '
            'and <tempo value="0.8">feel the nice activation</tempo>.'
        ),
        (
            '<tempo value="0.7">Blood heater coolers</tempo> are used to <tempo value="0.7">maintain '
            "thermo-regulation for patients on ecmo</tempo>. Heat exchange can occur via an "
            '<tempo value="0.7">oxygenator with an integrated heater</tempo> or via a '
            '<tempo value="0.7">stand-alone heat exchanger</tempo> integrated into the circuit. The '
            "heater fluid phase and the blood phase are never in direct contact."
        ),
        (
            'As found in our current <tempo value="0.7">adult brain-dead donor clinical practice '
            'guidelines</tempo>, the goal of fluid management is to <tempo value="0.7">maintain '
            "urinary output greater than or equal to 0.5 milliliters per kilogram per hour</tempo> and "
            'a <tempo value="0.7">CVP 4-12 millimeters of mercury</tempo>. This is in alignment with '
            'the <tempo value="0.7">DMG bundle</tempo>.'
        ),
        (
            'To provide <tempo value="0.5">precise and effective</tempo> care for various eye '
            'conditions, including <tempo value="0.5">hemmorrhagic conditions</tempo>, optometrists '
            "employ a variety of state-of-the-art diagnostic tools and methods, including "
            '<tempo value="0.5">Ocular Ultrasound, Fluorescein Angiography, and Optical Coherence '
            "Tomography</tempo>."
        ),
    )

    def __post_init__(self):
        REPORT_CARD_TEST_CASES = (
            self.SLURRING
            + self.ABBREVIATIONS_WITH_VOWELS
            + self.SLOW_SCRIPTS
            + self.HARD_SCRIPTS
            + self.HARD_SCRIPTS_2
            + self.QUESTIONS
            + self.QUESTIONS_WITH_VARIED_INFLECTION
            + self.QUESTIONS_WITH_UPWARD_INFLECTION
            + self.VARIOUS_INITIALISMS
        )
        REPORT_CARD_TEST_CASES = list(
            set(remove_xml(i) for i in REPORT_CARD_TEST_CASES)
        )
        random.shuffle(REPORT_CARD_TEST_CASES)
        self.REPORT_CARD_TEST_CASES = REPORT_CARD_TEST_CASES
