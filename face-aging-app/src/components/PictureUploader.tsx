import React, { useState } from "react";

interface Props {
  image: string;
  setImage: Function;
}

export default function PictureUploader({ image, setImage }: Props) {
  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    event.target.files != null
      ? setImage(URL.createObjectURL(event.target.files[0]))
      : setImage("");
  };

  return (
    <div>
      <input type="file" onChange={handleChange} />
      <img src={image} />
    </div>
  );
}
