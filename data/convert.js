const fs = require("fs");
const MidiConvert = require("midiconvert");
const util = require("util");

function convert(midiFileName, jsonFileName) {
    fs.readFile(midiFileName, "binary", function(err, midiBlob) {
        if(err) {
            throw err;
        }
        song_json = MidiConvert.parse(midiBlob);
        fs.writeFile(jsonFileName, JSON.stringify(song_json), "utf-8", function(err) {
            if(err) {
                throw err;
            }
        });
    });
};

convert(process.argv[2], process.argv[3])
