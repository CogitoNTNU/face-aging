import React, { useState } from "react";

import Header from "./Header";
import AgeSelector from "../components/AgeSelector";
import GenderSelector from "../components/GenderSelector";
import PictureUploader from "../components/PictureUploader";
import ModelHandler from "../components/ModelHandler";
import GeneratePictureButton from "../components/GeneratePictureButton";

export default function FaceAgeContainer() {
  const [targetAge, setTargetAge] = useState(3);
  const [targetGender, setTargetGender] = useState("1");
  const [image, setImage] = useState("");

  const registerTargetAge = (newTarget: number) => {
    setTargetAge(newTarget);
  };

  const registerTargetGender = (newTarget: string) => {
    setTargetGender(newTarget);
  };

  const registerImage = (objectUrl: string) => {
    setImage(objectUrl);
  };

  const generateFace = () => {};

  return (
    <div>
      <Header />
      <PictureUploader image={image} setImage={registerImage} />
      <ModelHandler />
      <AgeSelector
        label={"Select Age"}
        registerTargetAge={registerTargetAge}
        targetAge={targetAge}
      />
      <GenderSelector
        label={"Select Gender"}
        registerTargetGender={registerTargetGender}
        targetGender={targetGender}
      />
      <GeneratePictureButton generateFace={generateFace} />
    </div>
  );
}
