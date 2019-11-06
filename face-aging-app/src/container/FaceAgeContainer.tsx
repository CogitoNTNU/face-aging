import React from "react";

import Header from "./Header";
import AgeSelector from "../components/AgeSelector";
import GenderSelector from "../components/GenderSelector";
import PictureUploader from "../components/PictureUploader";
import PictureViewer from "../components/PictureViewer";
import GeneratePictureButton from "../components/GeneratePictureButton";

export default function FaceAgeContainer() {
  const [targetAge, setTargetAge] = React.useState(3);
  const [targetGender, setTargetGender] = React.useState("male");

  const registerTargetAge = (newTarget: number) => {
    setTargetAge(newTarget);
  };

  const registerTargetGender = (newTarget: string) => {
    setTargetGender(newTarget);
  };

  const generateFace = () => {};

  return (
    <div>
      <Header />
      <PictureUploader />
      <PictureViewer />
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
